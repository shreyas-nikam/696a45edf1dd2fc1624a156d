# QuLab: Lab 7: LLM Extraction with Streaming & Multi-Provider

## Building a Resilient Enterprise Knowledge Extractor with Adaptive LLM Routing

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

Welcome to the **OrgAIR Solutions Inc.'s AI Transformation Challenge**! This project, "QuLab: Lab 7," guides you through developing a next-generation "knowledge extraction" workflow using advanced LLM techniques. As a Software Developer at OrgAIR, you'll transform a fragile, single-LLM extraction pipeline into a robust, cost-effective, and secure system. The current setup struggles with reliability, unexpected costs, and a lack of real-time feedback, hindering OrgAIR's ability to deliver timely and accurate financial metrics, risk factors, and strategic initiatives to clients.

This Streamlit application serves as an interactive laboratory to demonstrate and explore key concepts for building production-ready LLM applications, focusing on resilience, efficiency, and safety.

## Features

This application showcases a comprehensive suite of features essential for enterprise LLM integration:

1.  **Environment Setup and Configuration**:
    *   Secure API key management (OpenAI, Anthropic).
    *   Dynamic daily budget configuration for LLM expenditures.
    *   Initialization of core LLM components (Model Router, Tool Caller, Guardrails).

2.  **LLM Routing & Fallbacks**:
    *   Intelligent multi-model router using `LiteLLM` to select the best LLM based on task type and availability.
    *   Automatic fallback mechanisms to alternative LLM providers in case of primary model failure (e.g., API issues, rate limits).
    *   Cost-aware routing decisions.

3.  **Real-time Streaming Extraction**:
    *   Implementation of asynchronous streaming capabilities for LLM responses.
    *   Provides token-by-token feedback during long document processing, enhancing user experience.

4.  **Native LLM Tool Calling**:
    *   Integration of native function calling, allowing LLMs to interact with custom internal tools (e.g., `OrgAIRCalculator`, `CompanyEvidenceDB`).
    *   Enables LLMs to perform complex, multi-step reasoning by orchestrating external data retrieval and calculations.
    *   Demonstration of LLM's thought process when calling tools.

5.  **Cost Management & Budget Enforcement**:
    *   Proactive tracking and enforcement of daily spending limits for LLM API usage.
    *   Prevents accidental overspending by blocking requests that would exceed the configured budget.

6.  **Input/Output Guardrails**:
    *   Implementation of robust safety guardrails for both input prompts and LLM outputs.
    *   **Input Guardrails**: Detects and mitigates prompt injection attacks, and enforces input length limits.
    *   **Output Guardrails**: Redacts sensitive Personally Identifiable Information (PII) like email addresses, phone numbers, SSNs, and credit card numbers from LLM responses.

7.  **Enhanced Logging**:
    *   Utilizes `structlog` for structured, human-readable, and machine-parsable logging across all LLM interactions, crucial for debugging, monitoring, and auditing.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   API keys for OpenAI and Anthropic (required for testing multi-provider features).

### Installation

1.  **Clone the repository** (assuming this code is part of a larger project or you'll create a local directory):

    ```bash
    git clone https://github.com/your-repo/qu-lab-7.git # Replace with actual repo URL
    cd qu-lab-7
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required dependencies**:
    Create a `requirements.txt` file in your project root with the following content:

    ```
    streamlit
    litellm
    structlog
    pydantic
    ```

    Then install them:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `source.py`**:
    The provided `app.py` imports a significant amount of core logic from a `source.py` file. You'll need to create this file with the necessary classes, functions, and configurations. While the full `source.py` content is not provided, its structure is implied by the `import` statements. It should contain:
    *   `Settings`, `TaskType`, `ModelConfig`, `MODEL_ROUTING`, `DailyBudget`, `ModelRouter`
    *   `simulate_failure_mode`, `synthetic_enterprise_document_text`, `streaming_document_text`
    *   `OrgAIRCalculator`, `CalculateOrgAIRInput`, `GetEvidenceInput`, `ProjectEBITDAInput`, `ToolDefinition`, `handle_calculate_org_air`, `handle_get_evidence`, `handle_project_ebitda`, `TOOLS`
    *   `OpenAINativeToolCaller`, `SafetyGuardrails`
    *   `logger` (structlog instance)
    *   `litellm` (configured instance)

    A basic example structure for `source.py` to allow `app.py` to run might look like:

    ```python
    # source.py
    import os
    import asyncio
    import json
    from decimal import Decimal
    from enum import Enum
    from typing import List, Dict, Any, Optional, AsyncGenerator

    import pydantic
    import structlog
    import litellm

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()

    # --- Dummy data for simulation ---
    synthetic_enterprise_document_text = """
    OrgAIR Solutions Inc. Financial Report 2023
    Revenue: $1.2 Billion
    Net Income: $150 Million
    Primary Risk Factor: Rapid technological obsolescence of legacy systems.
    Market Share: 15% in Q4 2023.
    """

    streaming_document_text = """
    This is a long document about the future of AI in finance.
    It covers various aspects including regulatory changes,
    the adoption of machine learning by banks, and the impact
    on traditional financial services. Key dates for innovation
    are Q1 2025 for new compliance frameworks, and H2 2026 for
    widespread blockchain integration. Company names mentioned
    include FinTech Innovations Corp., Global Bank Group, and
    AI Analytics Ltd. Market share projections suggest FinTech
    Innovations could capture an additional 5% by 2027.
    """

    # --- Pydantic Models and Enums ---
    class Settings(pydantic.BaseModel):
        OPENAI_API_KEY: Optional[str] = None
        ANTHROPIC_API_KEY: Optional[str] = None
        DAILY_COST_BUDGET_USD: Decimal = Decimal("1.00")
        DEBUG: bool = True

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    class TaskType(str, Enum):
        EVIDENCE_EXTRACTION = "evidence_extraction"
        CHAT_RESPONSE = "chat_response"
        TOOL_CALLING = "tool_calling" # Added for consistency

    class ModelConfig(pydantic.BaseModel):
        model_name: str
        provider: str
        task_types: List[TaskType]
        cost_per_1k_input_tokens: Decimal
        cost_per_1k_output_tokens: Decimal
        max_tokens: int

    MODEL_ROUTING: Dict[TaskType, List[ModelConfig]] = {
        TaskType.EVIDENCE_EXTRACTION: [
            ModelConfig(
                model_name="gpt-4o",
                provider="openai",
                task_types=[TaskType.EVIDENCE_EXTRACTION, TaskType.TOOL_CALLING],
                cost_per_1k_input_tokens=Decimal("0.005"),
                cost_per_1k_output_tokens=Decimal("0.015"),
                max_tokens=4096,
            ),
            ModelConfig(
                model_name="claude-3-5-sonnet",
                provider="anthropic",
                task_types=[TaskType.EVIDENCE_EXTRACTION],
                cost_per_1k_input_tokens=Decimal("0.003"),
                cost_per_1k_output_tokens=Decimal("0.015"),
                max_tokens=4096,
            ),
        ],
        TaskType.CHAT_RESPONSE: [
            ModelConfig(
                model_name="gpt-4o",
                provider="openai",
                task_types=[TaskType.CHAT_RESPONSE],
                cost_per_1k_input_tokens=Decimal("0.005"),
                cost_per_1k_output_tokens=Decimal("0.015"),
                max_tokens=4096,
            ),
            ModelConfig(
                model_name="claude-3-5-sonnet",
                provider="anthropic",
                task_types=[TaskType.CHAT_RESPONSE],
                cost_per_1k_input_tokens=Decimal("0.003"),
                cost_per_1k_output_tokens=Decimal("0.015"),
                max_tokens=4096,
            ),
        ],
        TaskType.TOOL_CALLING: [ # Ensure TOOL_CALLING is defined if used
            ModelConfig(
                model_name="gpt-4o",
                provider="openai",
                task_types=[TaskType.TOOL_CALLING],
                cost_per_1k_input_tokens=Decimal("0.005"),
                cost_per_1k_output_tokens=Decimal("0.015"),
                max_tokens=4096,
            ),
        ]
    }

    class DailyBudget(pydantic.BaseModel):
        limit_usd: Decimal
        spent_usd: Decimal = Decimal("0.00")
        last_reset_date: str = "" # To be implemented for actual daily reset

        def can_spend(self, estimated_cost_usd: Decimal) -> bool:
            return (self.spent_usd + estimated_cost_usd) <= self.limit_usd

        def record_spend(self, actual_cost_usd: Decimal):
            self.spent_usd += actual_cost_usd
            logger.info("budget_recorded", current_spend=f"{self.spent_usd:.4f}", cost_added=f"{actual_cost_usd:.4f}")

    _failure_modes = {}
    def simulate_failure_mode(model_name: str, enabled: bool):
        _failure_modes[model_name] = enabled
        logger.warning("failure_mode_simulation_toggle", model=model_name, enabled=enabled)

    class ModelRouter:
        def __init__(self):
            self.daily_budget = DailyBudget(limit_usd=Settings.DAILY_COST_BUDGET_USD)
            self.settings = Settings() # Assuming settings are loaded/updated by streamlit

        def _get_model_config(self, task: TaskType) -> List[ModelConfig]:
            return MODEL_ROUTING.get(task, [])

        async def _calculate_cost(self, model_config: ModelConfig, prompt_tokens: int, completion_tokens: int) -> Decimal:
            input_cost = (Decimal(prompt_tokens) / Decimal(1000)) * model_config.cost_per_1k_input_tokens
            output_cost = (Decimal(completion_tokens) / Decimal(1000)) * model_config.cost_per_1k_output_tokens
            return input_cost + output_cost

        async def complete(self, task: TaskType, messages: List[Dict[str, str]], stream: bool = False, tools: Optional[List[ToolDefinition]] = None) -> Any:
            model_configs = self._get_model_config(task)
            if not model_configs:
                raise RuntimeError(f"No models configured for task type: {task.value}")

            error_messages = []
            for model_config in model_configs:
                if _failure_modes.get(model_config.model_name, False):
                    logger.warning("llm_failure_simulated", model=model_config.model_name, task=task.value)
                    error_messages.append(f"Simulated failure for {model_config.model_name}")
                    continue

                try:
                    # Estimate token count for budget check (can be improved)
                    # For simplicity, let's just use a fixed number or simple heuristic for now
                    estimated_prompt_tokens = 100 # Example fixed estimate
                    estimated_cost = await self._calculate_cost(model_config, estimated_prompt_tokens, 0) # Only input cost for proactive check

                    if not self.daily_budget.can_spend(estimated_cost):
                        raise RuntimeError(f"Daily budget limit (${self.daily_budget.limit_usd:.2f}) exceeded. Current spend: ${self.daily_budget.spent_usd:.2f}")

                    logger.info("llm_request", model=model_config.model_name, task=task.value, stream=stream)
                    litellm_model_name = model_config.model_name
                    if model_config.provider == "anthropic":
                        litellm_model_name = f"anthropic/{model_config.model_name}"

                    tool_kwargs = {}
                    if tools:
                        tool_kwargs = {
                            "tools": [t.model_dump(mode='json') for t in tools],
                            "tool_choice": "auto"
                        }

                    if stream:
                        async def streaming_generator():
                            full_content = ""
                            response_generator = await litellm.acompletion(
                                model=litellm_model_name,
                                messages=messages,
                                stream=True,
                                **tool_kwargs
                            )
                            async for chunk in response_generator:
                                if chunk.choices and chunk.choices[0].delta.content:
                                    content_chunk = chunk.choices[0].delta.content
                                    full_content += content_chunk
                                    yield content_chunk
                                if chunk.usage: # usage might come in final chunk
                                    actual_input_tokens = chunk.usage.prompt_tokens
                                    actual_output_tokens = chunk.usage.completion_tokens
                                    actual_cost = await self._calculate_cost(model_config, actual_input_tokens, actual_output_tokens)
                                    self.daily_budget.record_spend(actual_cost)
                            logger.info("llm_stream_complete", model=model_config.model_name, final_content_length=len(full_content))
                        return streaming_generator()
                    else:
                        response = await litellm.acompletion(
                            model=litellm_model_name,
                            messages=messages,
                            **tool_kwargs
                        )
                        actual_input_tokens = response.usage.prompt_tokens
                        actual_output_tokens = response.usage.completion_tokens
                        actual_cost = await self._calculate_cost(model_config, actual_input_tokens, actual_output_tokens)
                        self.daily_budget.record_spend(actual_cost)
                        logger.info("llm_response", model=model_config.model_name, tokens_used=response.usage.total_tokens)
                        return response

                except Exception as e:
                    logger.warning("llm_fallback", model=model_config.model_name, task=task.value, error=str(e))
                    error_messages.append(f"Error with {model_config.model_name}: {e}")
            raise RuntimeError(f"All models failed for task type {task.value}: {'; '.join(error_messages)}")

        async def stream(self, task: TaskType, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
            async for chunk in await self.complete(task=task, messages=messages, stream=True):
                yield chunk

    class CalculateOrgAIRInput(pydantic.BaseModel):
        company_name: str
        innovation_score: int = pydantic.Field(ge=1, le=100)
        ai_adoption_rate: float = pydantic.Field(ge=0.0, le=1.0)

    class GetEvidenceInput(pydantic.BaseModel):
        company_name: str
        data_type: str

    class ProjectEBITDAInput(pydantic.BaseModel):
        company_name: str
        org_air_score: int
        projection_years: int

    class ToolDefinition(pydantic.BaseModel):
        type: str = "function"
        function: Dict[str, Any]

    TOOLS: Dict[str, ToolDefinition] = {
        "calculate_org_air": ToolDefinition(
            function={
                "name": "calculate_org_air",
                "description": "Calculates the Org-AI-R score for a company based on innovation and AI adoption.",
                "parameters": CalculateOrgAIRInput.model_json_schema(),
            }
        ),
        "get_evidence": ToolDefinition(
            function={
                "name": "get_evidence",
                "description": "Retrieves specific evidence or data points from the OrgAIR knowledge base.",
                "parameters": GetEvidenceInput.model_json_schema(),
            }
        ),
        "project_ebitda": ToolDefinition(
            function={
                "name": "project_ebitda",
                "description": "Projects the EBITDA impact for a company based on its Org-AI-R score over a period.",
                "parameters": ProjectEBITDAInput.model_json_schema(),
            }
        ),
    }

    # Mock tool handlers
    async def handle_calculate_org_air(company_name: str, innovation_score: int, ai_adoption_rate: float) -> Dict[str, Any]:
        logger.info("tool_call_mock", tool="calculate_org_air", company=company_name)
        org_air_score = (innovation_score * 0.7) + (ai_adoption_rate * 30)
        return {"org_air_score": round(org_air_score), "status": "success"}

    async def handle_get_evidence(company_name: str, data_type: str) -> Dict[str, Any]:
        logger.info("tool_call_mock", tool="get_evidence", company=company_name, data_type=data_type)
        mock_data = {
            "InnovateCorp": {
                "Org-AI-R Score": "78 (based on 2023 data)",
                "Financials": {"Revenue": "500M", "Profit": "50M"},
                "Risk Factors": ["Market Volatility", "Talent Shortage"],
            }
        }
        if company_name in mock_data and data_type in mock_data[company_name]:
            return {"evidence": mock_data[company_name][data_type], "status": "success"}
        return {"evidence": f"No {data_type} found for {company_name}.", "status": "not_found"}

    async def handle_project_ebitda(company_name: str, org_air_score: int, projection_years: int) -> Dict[str, Any]:
        logger.info("tool_call_mock", tool="project_ebitda", company=company_name)
        base_ebitda = 100_000_000 # hypothetical
        growth_rate = 1 + (org_air_score / 1000) # Small growth for higher scores
        projected_ebitda = base_ebitda * (growth_rate ** projection_years)
        return {"projected_ebitda": round(projected_ebitda, 2), "currency": "USD", "years": projection_years, "status": "success"}


    _tool_handlers = {
        "calculate_org_air": handle_calculate_org_air,
        "get_evidence": handle_get_evidence,
        "project_ebitda": handle_project_ebitda,
    }

    class OpenAINativeToolCaller:
        def __init__(self):
            self.model_router = ModelRouter() # Re-use the router
            self.tools_list = list(TOOLS.values()) # Tools in a format expected by LiteLLM

        async def chat_with_tools(self, messages: List[Dict[str, str]], model: str = "gpt-4o") -> Dict[str, Any]:
            logger.info("tool_caller_initiate", messages=messages)
            tool_calls = []
            final_response = ""
            current_messages = list(messages)

            try:
                # First call: LLM decides if it needs to call a tool
                response = await self.model_router.complete(task=TaskType.TOOL_CALLING, messages=current_messages, tools=self.tools_list)
                
                # Check for tool calls
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        logger.info("tool_call_detected", function=function_name, arguments=arguments)

                        if function_name in _tool_handlers:
                            tool_result = await _tool_handlers[function_name](**arguments)
                            tool_calls.append({"tool": function_name, "arguments": arguments, "result": tool_result})
                            
                            # Add tool message to conversation history
                            current_messages.append(response.choices[0].message)
                            current_messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(tool_result),
                            })
                            logger.info("tool_execution_complete", function=function_name, result=tool_result)
                        else:
                            tool_calls.append({"tool": function_name, "arguments": arguments, "result": {"error": "Tool not found"}})
                            current_messages.append(response.choices[0].message)
                            current_messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps({"error": "Tool not found"}),
                            })
                            logger.error("tool_not_found", function=function_name)

                    # Second call: LLM summarizes the tool results
                    final_response_obj = await self.model_router.complete(task=TaskType.TOOL_CALLING, messages=current_messages)
                    final_response = final_response_obj.choices[0].message.content
                else:
                    final_response = response.choices[0].message.content

            except Exception as e:
                logger.error("tool_caller_error", error=str(e), messages=current_messages)
                final_response = f"An error occurred: {e}"

            return {"response": final_response, "tool_calls": tool_calls}


    class SafetyGuardrails:
        def __init__(self):
            self.max_input_length = 5000
            self.prompt_injection_patterns = [
                r"ignore previous instructions",
                r"disregard all prior commands",
                r"forget everything",
                r"as an AI, you must",
                r"new persona",
                r"act as",
                r"jailbreak",
            ]
            self.pii_patterns = {
                "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
                "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
            }
            logger.info("safety_guardrails_initialized")

        async def validate_input(self, text: str) -> (bool, str, Optional[str]):
            # Length check
            if len(text) > self.max_input_length:
                logger.warning("input_guardrail_blocked", reason="length_exceeded", length=len(text))
                return False, text, f"Input exceeds maximum allowed length of {self.max_input_length} characters."

            # Prompt injection check
            for pattern in self.prompt_injection_patterns:
                if asyncio.run(self._contains_pattern(text, pattern)):
                    logger.warning("input_guardrail_blocked", reason="prompt_injection_detected", pattern=pattern)
                    return False, text, f"Potential prompt injection detected based on pattern: '{pattern}'."

            logger.info("input_guardrail_passed")
            return True, text, None

        async def validate_output(self, text: str) -> (bool, str):
            sanitized_text = text
            contains_pii = False
            for pii_type, pattern in self.pii_patterns.items():
                if asyncio.run(self._contains_pattern(text, pattern)):
                    contains_pii = True
                    sanitized_text = asyncio.run(self._redact_pattern(sanitized_text, pattern, f"[{pii_type.upper()}_REDACTED]"))
            
            if contains_pii:
                logger.warning("output_guardrail_redacted", pii_detected=True)
            else:
                logger.info("output_guardrail_passed", pii_detected=False)
            
            return not contains_pii, sanitized_text # Returns True if NO PII was found

        async def _contains_pattern(self, text: str, pattern: str) -> bool:
            import re
            return bool(re.search(pattern, text, re.IGNORECASE))

        async def _redact_pattern(self, text: str, pattern: str, replacement: str) -> str:
            import re
            return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Initialize settings (can be overridden by Streamlit inputs)
    _initial_settings = Settings()
    litellm.set_verbose = _initial_settings.DEBUG
    litellm.openai_key = _initial_settings.OPENAI_API_KEY
    litellm.anthropic_key = _initial_settings.ANTHROPIC_API_KEY
    ```

## Usage

1.  **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

    Your web browser will automatically open to the Streamlit application (usually `http://localhost:8501`).

2.  **Navigate and Configure**:
    *   Start on the "Application Overview" page to understand the project goals.
    *   Proceed to the "**1. Environment Setup**" page. Here, you **must enter your OpenAI and Anthropic API keys** and set a daily budget. Click "Configure LLM Environment" to initialize the backend components. Without this step, other features will be blocked.

3.  **Explore Features**:
    *   Use the sidebar navigation to visit each section:
        *   **2. LLM Routing & Fallbacks**: Test different task types, observe model routing, and experiment with simulating model failures to see fallbacks in action.
        *   **3. Real-time Streaming Extraction**: Enter a prompt and witness token-by-token LLM output, demonstrating improved user experience.
        *   **4. Native LLM Tool Calling**: Ask complex queries that require the LLM to call predefined internal tools (like calculating an Org-AI-R score or projecting EBITDA). Observe the LLM's reasoning and the tool outputs.
        *   **5. Cost Management & Budget Enforcement**: Make multiple LLM requests to exceed the intentionally low default daily budget and see the system proactively block further API calls.
        *   **6. Input/Output Guardrails**: Test various inputs, including prompt injection attempts and text containing PII, to see the guardrails detect and act on them.

4.  **Monitor Logs**:
    *   Each page includes an "Show Raw Logs" expander. Open it to view the detailed `structlog` output, which provides insights into model choices, fallbacks, budget checks, tool calls, and guardrail decisions.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── source.py               # Core LLM orchestration logic, models, tools, guardrails, budget
├── requirements.txt        # Python dependencies
└── README.md               # This README file
```

*   `app.py`: Contains the Streamlit frontend, session state management, and orchestration of calls to the `source.py` backend.
*   `source.py`: Encapsulates all the heavy lifting – `ModelRouter` for multi-provider routing and cost management, `OpenAINativeToolCaller` for function calling, `SafetyGuardrails` for input/output sanitization, Pydantic models for configuration and tool schemas, and utility functions for data and simulation.
*   `requirements.txt`: Lists all Python libraries needed for the project.

## Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Backend/LLM Orchestration**: [Python](https://www.python.org/), [LiteLLM](https://litellm.ai/), [Asyncio](https://docs.python.org/3/library/asyncio.html)
*   **Logging**: [Structlog](https://www.structlog.org/)
*   **Data Modeling/Validation**: [Pydantic](https://docs.pydantic.dev/latest/)
*   **LLM Providers**: [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/) (configurable)

## Contributing

This project is primarily a lab exercise. However, if you find issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file (if created) for details. A common choice for educational and open-source projects.

## Contact

For questions or feedback, please reach out to:

*   **QuantUniversity** - [Website](https://www.quantuniversity.com/)
*   **GitHub Issues**: [Project Issues](https://github.com/your-repo/qu-lab-7/issues) (Replace with actual repo URL)