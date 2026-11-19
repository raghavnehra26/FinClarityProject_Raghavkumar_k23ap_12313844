from ai_agent import _has_lending_decision_claim, answer_chat, answer_question, ollama_status


def _ready_status(model="llama3.1:8b"):
    return {
        "api_connected": True,
        "model_available": True,
        "model": model,
        "models": [model],
        "base_url": "http://localhost:11434",
        "message": "Connected",
    }


def test_local_fallback_uses_actual_assessment_without_approval_claims(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: {
        "api_connected": False, "model_available": False, "models": [],
        "model": None, "base_url": "http://localhost:11434",
    })
    result = {
        "prediction": 1,
        "prediction_label": "Defaulted",
        "positive_class_label": "Defaulted",
        "negative_class_label": "Did not default",
        "target": "Historical default for issued loans",
        "model_score": 0.73,
        "score_warning": "uncalibrated",
        "applicant_inputs": {"loan_amnt": 1000},
        "explanations": [{
            "feature": "loan_amnt", "label": "Requested loan amount",
            "value": "1000", "contribution": 0.12,
            "direction": "higher predicted default score",
        }],
    }
    answer, notice = answer_question(result, "Why?")
    assert notice is not None
    assert "0.730" in answer
    assert "Requested loan amount" in answer
    assert "12.0 percentage points" in answer
    assert "toward “Defaulted”" in answer
    assert "approval decision" in answer


def test_ollama_status_detects_exact_installed_tag(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setattr("ai_agent._get_json", lambda url: {
        "models": [{"name": "llama3.2:3b", "size": 2_000_000_000}]
    })
    status = ollama_status()
    assert status["api_connected"]
    assert status["model_available"]
    assert status["model"] == "llama3.2:3b"
    assert status["models"] == ["llama3.2:3b"]


def test_ollama_status_selects_only_detected_llama_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("ai_agent._get_json", lambda url: {
        "models": [{"name": "llama3.2:3b"}, {"name": "qwen3:8b"}]
    })
    status = ollama_status()
    assert status["api_connected"]
    assert status["model"] == "llama3.2:3b"
    assert status["model_available"]


def test_installed_preferred_llama_is_selected_automatically(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("ai_agent._get_json", lambda url: {
        "models": [{"name": "llama3.2:3b"}, {"name": "llama3.1:8b"}]
    })
    status = ollama_status()
    assert status["model"] == "llama3.1:8b"
    assert status["model_available"]


def test_empty_ollama_tag_list_does_not_claim_a_connected_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setattr("ai_agent._get_json", lambda url: {"models": []})
    status = ollama_status()
    assert status["api_connected"]
    assert not status["model_available"]
    assert status["model"] == "llama3.2:3b"


def test_ollama_tool_call_uses_registered_tools_and_real_evidence(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append((url, payload))
        if len(calls) == 1:
            return {"message": {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "get_model_performance", "arguments": {},
                }}],
            }}
        return {"message": {"content": "The saved test metrics are available in the evaluation."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    artifact = {
        "metadata": {"target_definition": "historical source membership"},
        "evaluation": {
            "selected_model": "Random Forest",
            "validation_average_precision": {"Random Forest": 0.92},
            "test": {"metrics": {"f1": 0.9}},
        },
    }
    answer, evidence, notice = answer_chat(
        "How did the model perform?", artifact=artifact
    )
    assert "saved test metrics" in answer
    assert "model metrics" in evidence
    assert "historical data" in answer
    assert notice is None
    assert calls[0][0].endswith("/api/chat")
    assert calls[0][1]["model"] == "llama3.1:8b"
    assert calls[0][1]["stream"] is False
    assert calls[0][1]["tools"]
    assert calls[1][1]["messages"][-1]["tool_name"] == "get_model_performance"
    assert "0.9" in calls[1][1]["messages"][-1]["content"]


def test_model_without_tool_call_uses_controlled_python_routing_then_llama(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"role": "assistant", "content": "I will check evidence."}}
        return {"message": {"content": "A debt-to-income ratio compares debt payments with income."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    answer, evidence, notice = answer_chat("What is debt-to-income?")
    assert "compares debt payments" in answer
    assert "financial education" in evidence
    assert notice is None
    assert "debt_to_income" in calls[1]["messages"][-2]["content"]
    assert "tools" not in calls[1]


def test_ollama_unavailable_uses_local_fallback_without_claiming_llama(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: {
        "api_connected": False, "model_available": False, "models": [],
        "model": None, "base_url": "http://localhost:11434",
    })
    monkeypatch.setattr("ai_agent._post_json", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call model")))
    answer, evidence, notice = answer_chat("What is a debt-to-income ratio?")
    assert "Debt To Income" in answer
    assert "Monthly debt obligations" in answer
    assert evidence == "general financial education"
    assert "Ollama is unavailable" in notice


def test_invalid_tool_call_arguments_are_rejected(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    monkeypatch.setattr("ai_agent._post_json", lambda *args, **kwargs: {
        "message": {"role": "assistant", "tool_calls": [{"function": {
            "name": "get_financial_knowledge", "arguments": {"code": "print(1)"},
        }}]}
    })
    answer, evidence, notice = answer_chat("What is debt-to-income?")
    assert "Debt To Income" in answer
    assert evidence == "general financial education"
    assert "expected format" in notice


def test_loan_amount_tool_accepts_model_generated_concept_argument():
    from agent_tools import build_tools, execute_tool
    result = execute_tool("explain_financial_concept", {"concept": "loan_amount"}, build_tools(None, None))
    assert result["available"]
    assert result["concept"] == "loan_amount"
    assert "principal requested or borrowed" in result["definition"]


def test_general_loan_and_greeting_questions_have_useful_local_answers(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: {
        "api_connected": False, "model_available": False, "models": [],
        "model": None, "base_url": "http://localhost:11434",
    })
    greeting, evidence, _ = answer_chat("hi")
    assert "FinClarity" in greeting
    assert evidence == "greeting"
    answer, evidence, _ = answer_chat("what is loan")
    assert "money borrowed" in answer
    assert evidence == "general financial education"


def test_financial_tool_missing_argument_is_inferred_from_question(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"role": "assistant", "tool_calls": [{"function": {
                "name": "explain_financial_concept", "arguments": {},
            }}]}}
        return {"message": {"content": "A loan is money borrowed under an agreement to repay it."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    answer, evidence, notice = answer_chat("what is loan")
    assert "money borrowed" in answer
    assert evidence == "financial education"
    assert notice is None
    assert '"concept": "loan"' in calls[1]["messages"][-1]["content"]


def test_general_question_uses_local_llm_without_finclarity_tools_or_disclaimer(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append(payload)
        return {"message": {"content": "Machine learning finds patterns in examples and uses them to make predictions or organize information."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    answer, evidence, notice = answer_chat("What is machine learning?")
    assert "finds patterns" in answer
    assert evidence == "general knowledge"
    assert notice is None
    assert len(calls) == 1
    assert "tools" not in calls[0]
    assert "historical data" not in answer


def test_general_technical_definition_does_not_trigger_model_metrics_tool():
    from ai_agent import _routed_tool_names
    assert _routed_tool_names("What is the difference between precision and recall?", False) == []
    assert _routed_tool_names("Explain Random Forest with a simple example", False) == []
    assert _routed_tool_names("How many accepted records are available?", False) == ["get_dataset_statistics"]


def test_agent_executes_multiple_tool_rounds_and_returns_results_to_llama(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"role": "assistant", "tool_calls": [{"function": {
                "name": "get_current_shap_explanation", "arguments": {},
            }}]}}
        if len(calls) == 2:
            assert calls[1]["messages"][-1]["tool_name"] == "get_current_shap_explanation"
            return {"message": {"role": "assistant", "tool_calls": [{"function": {
                "name": "get_global_feature_importance", "arguments": {},
            }}]}}
        assert calls[2]["messages"][-1]["tool_name"] == "get_global_feature_importance"
        return {"message": {"content": "Local SHAP explains this profile; global importance summarizes the evaluation sample."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    result = {"prediction_label": "Accepted-file record", "target": "source membership", "explanations": [{"label": "Amount", "contribution": 0.1}]}
    artifact = {"evaluation": {"global_feature_importance": [{"label": "Amount", "importance_mean": 0.2}]}}
    answer, evidence, notice = answer_chat("Compare my SHAP explanation with overall feature importance", result=result, artifact=artifact)
    assert "global importance" in answer
    assert evidence == "SHAP evidence, permutation importance"
    assert notice is None
    assert len(calls) == 3


def test_conversation_history_is_passed_to_local_model(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    seen = []

    def fake_post(url, payload, timeout=120):
        seen.append(payload)
        if len(seen) == 1:
            return {"message": {"role": "assistant", "tool_calls": [{
                "function": {"name": "get_financial_knowledge", "arguments": {}}
            }]}}
        return {"message": {"content": "Here is the definition."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    answer_chat("Can you explain more?", [{"role": "user", "content": "What is DTI?"}])
    assert seen[0]["messages"][1] == {"role": "user", "content": "What is DTI?"}


def test_lending_decision_paraphrase_is_rejected():
    assert _has_lending_decision_claim(
        "The model makes a lender less likely to accept the loan."
    )
    assert _has_lending_decision_claim(
        "This means you are more likely to get approved."
    )
    assert _has_lending_decision_claim(
        "The model was less likely to classify the loan as accepted."
    )
    assert not _has_lending_decision_claim(
        "The profile received a lower accepted-file membership score."
    )


def test_unsafe_live_model_wording_is_replaced_with_local_evidence(monkeypatch):
    monkeypatch.setattr("ai_agent.ollama_status", lambda model=None: _ready_status())
    calls = []

    def fake_post(url, payload, timeout=120):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"role": "assistant", "tool_calls": [{
                "function": {"name": "get_shap_explanation", "arguments": {}}
            }]}}
        return {"message": {"content": "The model was less likely to classify the loan as accepted."}}

    monkeypatch.setattr("ai_agent._post_json", fake_post)
    answer, evidence, notice = answer_chat("What do my SHAP contributions show?")
    assert evidence == "SHAP evidence"
    assert "loan as accepted" not in answer
    assert "lending-safety check" in notice
