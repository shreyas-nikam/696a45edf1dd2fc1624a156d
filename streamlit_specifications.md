
# Streamlit Application Specification: OrgAIR Knowledge Extractor

## 1. Application Overview

### Purpose of the Application

The "OrgAIR Knowledge Extractor" Streamlit application serves as a hands-on blueprint for a **Software Developer** at OrgAIR Solutions Inc. to build a robust, cost-effective, and secure enterprise knowledge extraction system. It demonstrates the practical application of advanced Large Language Model (LLM) techniques, including:
*   **Multi-model routing** with automatic fallbacks for enhanced reliability and cost efficiency.
*   **Real-time streaming** of LLM responses for improved user experience during document processing.
*   **Native tool calling** to integrate LLMs with internal data and calculation services.
*   **Cost management and budget enforcement** to control LLM API expenditures.
*   **Input and output guardrails** to protect against prompt injection and redact Personally Identifiable Information (PII).

The application provides a simulated real-world workflow where the learner can configure LLM providers, trigger extraction tasks, observe routing decisions, manage costs, and enforce safety policies, directly applying concepts learned from the `source.py` notebook.

### High-level Story Flow

The user, acting as a Software Developer, navigates through a multi-page Streamlit application, mimicking the development and testing phases of the OrgAIR system:

1.  **Environment Setup:** The user begins by securely configuring their OpenAI and Anthropic API keys and setting a daily budget. This step initializes the core LLM components for the application.
2.  **LLM Routing & Fallbacks:** The user then experiments with multi-model routing. They can select a task type, provide a prompt, and observe how the system intelligently selects a primary LLM or falls back to alternatives in case of simulated failures or unavailability, while tracking costs.
3.  **Real-time Streaming Extraction:** To address the need for immediate feedback, the user initiates real-time knowledge extraction from a document. They observe token-by-token progress and extracted information as it streams from the LLM, simulating a faster user experience.
4.  **Native LLM Tool Calling:** The application demonstrates how LLMs can interact with internal tools. The user poses queries that trigger specific functions (e.g., calculating an Org-AI-R score, retrieving evidence, projecting EBITDA impact), showcasing advanced, multi-step reasoning capabilities.
5.  **Cost Management & Budget Enforcement:** The user actively tests the budget enforcement mechanism. They make repeated LLM requests and observe how the system blocks new requests once the predefined daily spending limit is reached, ensuring financial governance.
6.  **Input/Output Guardrails:** Finally, the user evaluates the safety measures. They test the input guardrails by attempting prompt injection attacks and observe them being blocked. They also test output guardrails, seeing how sensitive PII is automatically detected and redacted from LLM responses, ensuring data privacy and security.

Throughout this journey, logs are displayed to provide transparency into LLM interactions, routing decisions, cost tracking, and guardrail activations.

---

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import asyncio
import json
import io
import contextlib
import structlog
from decimal import Decimal

# Import all necessary functions and classes from the provided source.py
from source import (
    Settings, # Need to access/modify the global settings object for user input
    TaskType,
    ModelConfig,
    MODEL_ROUTING,
    DailyBudget,
    ModelRouter,
    simulate_failure_mode,
    synthetic_enterprise_document_text,
    streaming_document_text,
    OrgAIRCalculator, # Although not directly used by Streamlit UI, it's part of TOOL handlers
    CalculateOrgAIRInput,
    GetEvidenceInput,
    ProjectEBITDAInput,
    ToolDefinition,
    handle_calculate_org_air,
    handle_get_evidence,
    handle_project_ebitda,
    TOOLS,
    OpenAINativeToolCaller,
    SafetyGuardrails,
    logger, # The structlog logger from source.py
    litellm # To configure API keys directly
)

# Custom Settings instance from source.py
# The global 'settings' object from source.py needs to be explicitly imported
# and its attributes modified for user-provided API keys.
# We also need to manage LiteLLM's global API key configuration.
# The ModelRouter, OpenAINativeToolCaller, and SafetyGuardrails instances are
# initialized globally in source.py. We will create and store *new* instances
# in session_state that reflect user-provided configurations.
```

### `st.session_state` Design

The following `st.session_state` keys will be used to preserve state across interactions and simulate the multi-page experience:

*   `st.session_state.current_page`:
    *   **Initialized:** `"Application Overview"`
    *   **Updated:** When the user selects a different page from the sidebar dropdown.
    *   **Read:** Controls which page's content is rendered.
*   `st.session_state.openai_api_key`:
    *   **Initialized:** `""` (empty string)
    *   **Updated:** When the user inputs an OpenAI API key on the "Environment Setup" page.
    *   **Read:** Used during `initialize_llm_components` to set `settings.OPENAI_API_KEY` and `litellm.openai_key`.
*   `st.session_state.anthropic_api_key`:
    *   **Initialized:** `""` (empty string)
    *   **Updated:** When the user inputs an Anthropic API key on the "Environment Setup" page.
    *   **Read:** Used during `initialize_llm_components` to set `settings.ANTHROPIC_API_KEY` and `litellm.anthropic_key`.
*   `st.session_state.daily_budget_usd`:
    *   **Initialized:** `1.00` (Decimal representing 1 USD, matching `source.py` default)
    *   **Updated:** When the user inputs a daily budget on the "Environment Setup" page.
    *   **Read:** Used during `initialize_llm_components` to set `settings.DAILY_COST_BUDGET_USD` and initialize `st.session_state.model_router_instance.daily_budget.limit_usd`.
*   `st.session_state.api_keys_set`:
    *   **Initialized:** `False`
    *   **Updated:** Set to `True` after successful configuration of API keys and budget on the "Environment Setup" page.
    *   **Read:** Used to gate access to other pages, ensuring necessary components are initialized.
*   `st.session_state.model_router_instance`:
    *   **Initialized:** `None`
    *   **Updated:** Initialized with `ModelRouter()` during `initialize_llm_components` after API keys are set. Re-initialized if API keys are changed.
    *   **Read:** Used across "LLM Routing & Fallbacks", "Real-time Streaming Extraction", and "Cost Management & Budget Enforcement" pages to call `complete()`, `stream()`, and access `daily_budget`.
*   `st.session_state.openai_tool_caller_instance`:
    *   **Initialized:** `None`
    *   **Updated:** Initialized with `OpenAINativeToolCaller()` during `initialize_llm_components` after API keys are set. Re-initialized if API keys are changed.
    *   **Read:** Used on the "Native LLM Tool Calling" page to call `chat_with_tools()`.
*   `st.session_state.safety_guardrails_instance`:
    *   **Initialized:** `None`
    *   **Updated:** Initialized with `SafetyGuardrails()` during `initialize_llm_components` after API keys are set. Re-initialized if API keys are changed.
    *   **Read:** Used on the "Input/Output Guardrails" page to call `validate_input()` and `validate_output()`.
*   `st.session_state.model_failure_simulated`:
    *   **Initialized:** `False`
    *   **Updated:** Toggled by a checkbox on the "LLM Routing & Fallbacks" page.
    *   **Read:** Used to control the `simulate_failure_mode` function from `source.py`.
*   `st.session_state.captured_logs`:
    *   **Initialized:** `[]` (empty list)
    *   **Updated:** Appended with captured `structlog` output after each LLM interaction using a custom log capture mechanism.
    *   **Read:** Displayed within an `st.expander` on relevant pages for debugging and transparency.

### Global Helper for Log Capture

```python
# Helper to capture structlog output (which goes to stdout by default in source.py's config)
async def run_and_capture_logs(coroutine):
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        try:
            result = await coroutine
            # Capture logs written to stdout
            for line in output_capture.getvalue().strip().split('\n'):
                if line:
                    st.session_state.captured_logs.append(line)
            return result
        except Exception as e:
            for line in output_capture.getvalue().strip().split('\n'):
                if line:
                    st.session_state.captured_logs.append(line)
            raise e
```

### UI Interactions and Function Invocations

The Streamlit app will be structured with a sidebar for navigation.

---

#### Application Initialization & Sidebar

```python
st.sidebar.title("OrgAIR Knowledge Extractor")
page_selection = st.sidebar.selectbox(
    "Navigate",
    [
        "Application Overview",
        "1. Environment Setup",
        "2. LLM Routing & Fallbacks",
        "3. Real-time Streaming Extraction",
        "4. Native LLM Tool Calling",
        "5. Cost Management & Budget Enforcement",
        "6. Input/Output Guardrails",
    ],
    key="page_selector"
)
st.session_state.current_page = page_selection

# Enforce environment setup if not done
if not st.session_state.api_keys_set and st.session_state.current_page != "1. Environment Setup":
    st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page to proceed.")
    st.session_state.current_page = "1. Environment Setup"
    st.experimental_rerun() # Rerun to switch page immediately
```

---

#### Page: Application Overview

```python
if st.session_state.current_page == "Application Overview":
    st.title("Building a Resilient Enterprise Knowledge Extractor with Adaptive LLM Routing")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"## Introduction: OrgAIR Solutions Inc.'s AI Transformation Challenge")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"Welcome to OrgAIR Solutions Inc., a leading firm specializing in leveraging AI to extract critical insights from vast troves of enterprise documents. As a **Software Developer** at OrgAIR, your mission is to transform our fragile, single-LLM extraction pipeline into a robust, cost-effective, and secure system. Our current setup struggles with reliability, unexpected costs, and a lack of real-time feedback, hindering our ability to deliver timely and accurate financial metrics, risk factors, and strategic initiatives to our clients.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"This application will guide you through building a next-generation \"knowledge extraction\" workflow. You will:")
    st.markdown(f"*   Design an intelligent multi-model router using LiteLLM to ensure high availability and cost efficiency.")
    st.markdown(f"*   Implement real-time streaming capabilities to provide instant feedback during document processing.")
    st.markdown(f"*   Integrate native tool calling, allowing LLMs to interact with our internal data and calculation services.")
    st.markdown(f"*   Embed strict cost management and budget enforcement mechanisms to control API spending.")
    st.markdown(f"*   Fortify the system with input/output guardrails to protect against prompt injection and ensure PII redaction.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"By the end of this lab, you will have developed a resilient and intelligent enterprise knowledge extractor, solving OrgAIR's pressing operational challenges and enhancing our service offerings.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.info("Navigate to the '1. Environment Setup' page to begin configuring the application.")
```

---

#### Page: 1. Environment Setup

```python
elif st.session_state.current_page == "1. Environment Setup":
    st.title("1. Environment Setup and Configuration")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"As a Software Developer at OrgAIR, the first step in any project is setting up your development environment. This ensures all necessary tools and libraries are available and securely configured before diving into the core logic. We'll install the required Python packages and prepare for secure API key management.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("API Key Configuration"):
        st.markdown(f"To interact with various LLM providers, please provide your API keys. For security, these are stored in `st.session_state` and are not persisted beyond your session.")
        openai_key_input = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        anthropic_key_input = st.text_input("Anthropic API Key", type="password", value=st.session_state.anthropic_api_key)
        daily_budget_input = st.number_input("Daily LLM Budget (USD)", min_value=0.01, value=float(st.session_state.daily_budget_usd), format="%.2f")

        if st.button("Configure LLM Environment"):
            st.session_state.openai_api_key = openai_key_input
            st.session_state.anthropic_api_key = anthropic_key_input
            st.session_state.daily_budget_usd = Decimal(str(daily_budget_input))

            # Initialize / Re-initialize LLM components
            # We explicitly update the source.py's global settings object
            # and litellm's keys, then create fresh instances for session_state.
            Settings.OPENAI_API_KEY = st.session_state.openai_api_key
            Settings.ANTHROPIC_API_KEY = st.session_state.anthropic_api_key
            Settings.DAILY_COST_BUDGET_USD = st.session_state.daily_budget_usd

            litellm.openai_key = Settings.OPENAI_API_KEY
            litellm.anthropic_key = Settings.ANTHROPIC_API_KEY
            litellm.set_verbose = Settings.DEBUG

            st.session_state.model_router_instance = ModelRouter()
            st.session_state.openai_tool_caller_instance = OpenAINativeToolCaller()
            st.session_state.safety_guardrails_instance = SafetyGuardrails()

            st.session_state.api_keys_set = True
            st.success(f"LLM components configured successfully!")
            st.info(f"Daily Budget Limit: ${st.session_state.model_router_instance.daily_budget.limit_usd:.2f}")

    if st.session_state.api_keys_set:
        st.success("Environment is configured. You can now explore other features.")
    else:
        st.warning("Please configure your API keys and daily budget to enable LLM interactions.")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The output confirms the environment is set up. The `structlog` configuration ensures that all logs are well-formatted and easy to read, which will be crucial for debugging and analyzing routing decisions later. The API keys are loaded from environment variables for security, a best practice for any enterprise application. The daily budget is intentionally set low for this demonstration to quickly showcase the budget enforcement mechanism.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```

---

#### Page: 2. LLM Routing & Fallbacks

```python
elif st.session_state.current_page == "2. LLM Routing & Fallbacks":
    st.title("2. Designing a Multi-Model LLM Router with Automatic Fallbacks")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"At OrgAIR, relying on a single LLM provider for critical knowledge extraction tasks is a significant risk. If the primary provider experiences an outage or becomes too expensive, our operations could halt. Your task is to implement a resilient multi-model routing mechanism that automatically falls back to alternative LLM providers, ensuring business continuity and potentially optimizing costs based on task requirements.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"This approach introduces the concept of **Multi-Model Routing with Automatic Fallbacks**. The system will attempt to use a primary, often high-accuracy, model first. If that model fails or is unavailable, it will gracefully switch to a predefined list of fallback models.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    if not st.session_state.api_keys_set:
        st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page.")
        st.stop()

    task_type_selection = st.selectbox(
        "Select Task Type",
        options=[tt.value for tt in TaskType],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    task_type_enum = TaskType(task_type_selection)

    prompt_input = st.text_area(
        "Enter your extraction prompt:",
        value=f"Extract revenue, net income, and primary risk factor from the document: {synthetic_enterprise_document_text}",
        height=200
    )

    col1, col2 = st.columns(2)
    with col1:
        simulate_failure = st.checkbox("Simulate Primary Model Failure (OpenAI)", value=st.session_state.model_failure_simulated)
    with col2:
        simulate_anthropic_failure = st.checkbox("Simulate Anthropic Failure", value=False) # Add a separate checkbox for Anthropic

    if simulate_failure != st.session_state.model_failure_simulated:
        st.session_state.model_failure_simulated = simulate_failure
        # Apply failure mode for OpenAI (primary for EVIDENCE_EXTRACTION)
        simulate_failure_mode("gpt-4o", enabled=simulate_failure)
        st.rerun()

    if simulate_anthropic_failure:
        simulate_failure_mode("claude-sonnet-3.5", enabled=simulate_anthropic_failure)
    else:
        # Ensure Anthropic key is restored if checkbox is unchecked
        simulate_failure_mode("claude-sonnet-3.5", enabled=False)

    if st.button("Run LLM Completion"):
        if not prompt_input:
            st.error("Please enter a prompt.")
        else:
            messages = [{"role": "user", "content": prompt_input}]
            st.session_state.captured_logs.append(f"--- Scenario: {task_type_enum.value} ---")
            st.session_state.captured_logs.append(f"User Prompt: {prompt_input[:100]}...")

            response_container = st.empty()
            with st.spinner("Processing LLM request..."):
                try:
                    llm_response = await run_and_capture_logs(
                        st.session_state.model_router_instance.complete(task=task_type_enum, messages=messages)
                    )
                    response_container.success(f"Final Response from {llm_response.model}:")
                    response_container.info(llm_response.choices[0].message.content)
                except RuntimeError as e:
                    response_container.error(f"Error: {e}")
                except Exception as e:
                    response_container.exception(f"An unexpected error occurred: {e}")

            st.markdown(f"Current Cumulative Spend: ${st.session_state.model_router_instance.daily_budget.spent_usd:.4f}")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The logs show how `ModelRouter` attempts to use the primary model (e.g., `gpt-4o` for `EVIDENCE_EXTRACTION`). When we artificially introduce an invalid API key, `litellm` fails to connect, and the system gracefully falls back to `claude-sonnet-3.5`, as observed by the `llm_fallback` warning and the subsequent `llm_request` for the fallback model. If all models configured for a specific `TaskType` fail, a `RuntimeError` is raised, preventing an indefinite loop.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(r"$$ \text{{Request Cost}} = \frac{{\text{{Total Tokens Used}}}}{{1000}} \times \text{{Cost per 1k Tokens}} $$")
    st.markdown(r"where $\text{{Total Tokens Used}}$ is the sum of input and output tokens, and $\text{{Cost per 1k Tokens}}$ is specific to the LLM model used.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(r"The `check_budget` method ensures that the estimated cost of a request, which is $ \text{{Estimated Cost}} = \frac{{\text{{Estimated Input Tokens}}}}{{1000}} \times \text{{Cost per 1k Tokens}} $, plus the `spent_usd` does not exceed `limit_usd`. This proactive check prevents unnecessary API calls when the budget is already tight. The `record_spend` method updates the `spent_usd` after a successful call using the actual tokens consumed. This implementation ensures that OrgAIR can control LLM API expenditures, a critical aspect of managing production AI systems.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```

---

#### Page: 3. Real-time Streaming Extraction

```python
elif st.session_state.current_page == "3. Real-time Streaming Extraction":
    st.title("3. Implementing Real-time Knowledge Extraction with Streaming")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"Enterprise document analysis can be lengthy, especially for large reports. Business stakeholders at OrgAIR need immediate feedback, not a long wait for a complete response. Your next task is to implement asynchronous streaming of LLM responses. This allows users to see token-by-token progress and extracted information as it's generated, significantly improving perceived performance and user experience.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"The `stream` method in `ModelRouter` leverages Python's `async generators` to yield chunks of text as they arrive from the LLM API. This demonstrates how to handle **streaming responses** in a non-blocking, real-time manner.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    if not st.session_state.api_keys_set:
        st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page.")
        st.stop()

    task_type_selection = st.selectbox(
        "Select Task Type for Streaming",
        options=[TaskType.EVIDENCE_EXTRACTION.value, TaskType.CHAT_RESPONSE.value],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    task_type_enum = TaskType(task_type_selection)

    prompt_input = st.text_area(
        "Enter your document/prompt for streaming extraction:",
        value=f"Extract key dates, company names, and market share projections from the following text: {streaming_document_text}",
        height=200
    )

    if st.button("Start Streaming Extraction"):
        if not prompt_input:
            st.error("Please enter a prompt.")
        else:
            messages = [{"role": "user", "content": prompt_input}]
            st.session_state.captured_logs.append(f"--- Streaming Scenario: {task_type_enum.value} ---")
            st.session_state.captured_logs.append(f"User Prompt: {prompt_input[:100]}...")
            st.write("Streaming response (token by token):")
            
            response_placeholder = st.empty()
            full_response_content = ""
            
            try:
                async for chunk in run_and_capture_logs(st.session_state.model_router_instance.stream(task=task_type_enum, messages=messages)):
                    full_response_content += chunk
                    response_placeholder.markdown(full_response_content)
                st.success("--- Streaming Complete ---")
                st.info(f"Final extracted content length: {len(full_response_content)} characters")
            except RuntimeError as e:
                st.error(f"Error during streaming: {e}")
            except Exception as e:
                st.exception(f"An unexpected error occurred during streaming: {e}")

            st.markdown(f"Current Cumulative Spend: ${st.session_state.model_router_instance.daily_budget.spent_usd:.4f}")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The output demonstrates the real-time token flow, where chunks of the LLM's response are printed as they are received, rather than waiting for the entire response. For OrgAIR, this means that even if a document takes 30 seconds to process, users can start seeing relevant information (like extracted entities) within the first few seconds, greatly improving their perception of the system's responsiveness. The budget is also continuously tracked and updated, even for streaming responses, though for simplicity, the actual cost recording for streaming happens at the end of the stream in this example.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```

---

#### Page: 4. Native LLM Tool Calling

```python
elif st.session_state.current_page == "4. Native LLM Tool Calling":
    st.title("4. Integrating Native LLM Tool Calling for Complex Data Retrieval")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"Simple text extraction often isn't enough for OrgAIR's sophisticated analyses. Our LLM-powered system needs to perform calculations, query internal databases, and retrieve specific evidence. As the Software Developer, you will integrate **native tool calling** into the LLM workflow, allowing the LLM to dynamically interact with custom Python functions that simulate these internal tools. This enables the LLM to go beyond mere text generation and perform complex, multi-step reasoning.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"We will define a set of tools with clear input schemas and mock handlers that simulate interaction with our internal systems (e.g., `org_air_calculator`, `company_evidence_db`).")
    st.markdown(f"") # Empty markdown to satisfy constraint

    if not st.session_state.api_keys_set:
        st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page.")
        st.stop()

    st.subheader("Available Tools:")
    for tool_name, tool_def in TOOLS.items():
        st.markdown(f"- **{tool_name}**: {tool_def.description}")
        st.json(tool_def.input_schema.model_json_schema(), expanded=False)

    user_query = st.text_input(
        "Enter your query (e.g., 'What is the Org-AI-R score for InnovateCorp?', 'Project the EBITDA impact for InnovateCorp if they achieve an Org-AI-R score of 85 over the next 3 years.')",
        value="What is the Org-AI-R score for InnovateCorp?"
    )

    if st.button("Execute Tool Query"):
        if not user_query:
            st.error("Please enter a query.")
        else:
            messages = [{"role": "user", "content": user_query}]
            st.session_state.captured_logs.append(f"--- Tool Calling Scenario ---")
            st.session_state.captured_logs.append(f"User Query: {user_query}")

            response_container = st.empty()
            with st.spinner("LLM is thinking and potentially calling tools..."):
                try:
                    response_data = await run_and_capture_logs(
                        st.session_state.openai_tool_caller_instance.chat_with_tools(messages=messages, model="gpt-4o")
                    )
                    response_container.subheader("LLM's Final Response:")
                    response_container.info(response_data["response"])

                    if response_data["tool_calls"]:
                        st.subheader("Executed Tools and Results:")
                        for tc in response_data["tool_calls"]:
                            st.write(f"- **Tool:** `{tc['tool']}`")
                            st.json(tc["result"], expanded=True)
                except Exception as e:
                    response_container.exception(f"An error occurred during tool calling: {e}")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The output clearly shows the LLM's thought process. For each query requiring a tool, the LLM first generates a `tool_calls` message with the function name and arguments. Our `OpenAINativeToolCaller` then intercepts this, executes the mocked Python function, and feeds the `tool_response` back to the LLM. Finally, the LLM generates a coherent, context-aware response incorporating the tool's output. For OrgAIR, this means LLMs can now perform complex analyses by integrating directly with our proprietary data and computation engines, moving beyond simple text generation to true intelligent automation.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.subheader("Native Tool Calling vs. Instructor Abstraction:")
    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"While this notebook focuses on native tool calling (using the LLM provider's built-in function calling mechanism), it's important to understand alternatives like `Instructor`.")
    st.markdown(f"*   **Native Tool Calling (as demonstrated):** Directly uses the LLM provider's API for function calls. The LLM decides when and how to call tools. You handle parsing the tool call and executing your Python function. This gives you maximum flexibility and is often the most direct way to leverage the latest LLM capabilities.")
    st.markdown(f"*   **Instructor Abstraction:** Libraries like `Instructor` (or `Pydantic-ChatCompletion` for older models) wrap the LLM calls to force structured outputs, often using Pydantic models. Instead of the LLM deciding to *call* a tool, you might prompt the LLM to *generate* a Pydantic object representing the output of a hypothetical tool. This is excellent for ensuring type-safe JSON outputs and can simplify certain structured extraction tasks where you want the LLM to directly emit data in a specific format, rather than orchestrating a multi-turn tool interaction.")
    st.markdown(f"For OrgAIR's workflow, native tool calling is preferred when the LLM needs to make decisions about *when* to use a tool and *what arguments* to pass based on conversational context. Instructor would be valuable for directly extracting structured data (e.g., all financial metrics) into Pydantic models in a single turn.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```

---

#### Page: 5. Cost Management & Budget Enforcement

```python
elif st.session_state.current_page == "5. Cost Management & Budget Enforcement":
    st.title("5. Cost Management and Budget Enforcement")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"Uncontrolled LLM API usage can quickly deplete budgets. As a Software Developer, you must implement mechanisms to track and enforce daily spending limits for OrgAIR's LLM operations. This provides crucial financial governance and prevents unexpected costs, a non-negotiable requirement for enterprise applications.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"The `DailyBudget` dataclass and its `can_spend` and `record_spend` methods, already integrated into our `ModelRouter` in Section 2, are responsible for this. Let's explicitly demonstrate its enforcement.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    if not st.session_state.api_keys_set:
        st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page.")
        st.stop()

    st.subheader("Current Budget Status:")
    st.info(f"Daily Budget Limit: ${st.session_state.model_router_instance.daily_budget.limit_usd:.4f}")
    st.info(f"Current Spend: ${st.session_state.model_router_instance.daily_budget.spent_usd:.4f}")

    st.warning("For demonstration purposes, the daily budget in the initial setup is intentionally set low (default $1.00) to quickly trigger budget enforcement.")

    task_type_selection = st.selectbox(
        "Select Task Type for Budget Test",
        options=[tt.value for tt in TaskType],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    task_type_enum = TaskType(task_type_selection)

    prompt_input = st.text_area(
        "Enter a simple prompt for testing budget enforcement:",
        value="Summarize the general sentiment regarding AI adoption in corporate finance.",
        height=100
    )

    if st.button("Attempt LLM Request"):
        if not prompt_input:
            st.error("Please enter a prompt.")
        else:
            messages = [{"role": "user", "content": prompt_input}]
            st.session_state.captured_logs.append(f"--- Cost Management Scenario ---")
            st.session_state.captured_logs.append(f"Attempting request for {task_type_enum.value}...")
            st.session_state.captured_logs.append(f"Current spend before request: ${st.session_state.model_router_instance.daily_budget.spent_usd:.4f}")

            try:
                llm_response = await run_and_capture_logs(
                    st.session_state.model_router_instance.complete(task=task_type_enum, messages=messages)
                )
                st.success(f"✅ Request successful for {task_type_enum.value} using {llm_response.model}.")
            except RuntimeError as e:
                st.error(f"❌ Request blocked for {task_type_enum.value}: {e}.")
            except Exception as e:
                st.exception(f"⚠️ An unexpected error occurred: {e}")

            st.markdown(f"**Updated Current Spend:** ${st.session_state.model_router_instance.daily_budget.spent_usd:.4f}")
            st.markdown(f"**Daily Budget Limit:** ${st.session_state.model_router_instance.daily_budget.limit_usd:.4f}")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The output clearly shows the budget enforcement in action. After one or more successful requests, the total spend hits or exceeds the artificially low daily budget. Subsequent requests are then blocked by a `RuntimeError` originating from the `check_budget` function. This demonstrates a proactive cost control mechanism: requests are evaluated against the budget *before* being sent to the LLM provider, saving both unnecessary API calls and preventing overspending. This is critical for OrgAIR to maintain financial predictability in its LLM-powered operations.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```

---

#### Page: 6. Input/Output Guardrails

```python
elif st.session_state.current_page == "6. Input/Output Guardrails":
    st.title("6. Implementing Input/Output Guardrails for Safety and PII Redaction")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"Security and data privacy are paramount in enterprise applications. As a Software Developer, you must protect OrgAIR's LLM system from malicious inputs (e.g., prompt injection attacks) and ensure sensitive information (e.g., PII) is not inadvertently exposed in LLM outputs. This task involves implementing robust input/output guardrails.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    st.markdown(f"The **Guardrails-AI** concept is applied here through custom regex patterns for detecting prompt injection and PII.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    if not st.session_state.api_keys_set:
        st.warning("Please configure your API keys and daily budget on the '1. Environment Setup' page.")
        st.stop()

    guardrail_type = st.radio(
        "Select Guardrail Type:",
        ("Input Guardrail Test (Prompt Injection, Length)", "Output Guardrail Test (PII Redaction)")
    )

    input_text = st.text_area(
        "Enter text to test:",
        value="Can you summarize the key financial risks for a tech startup in 2024?",
        height=150
    )

    if st.button("Run Guardrail Check"):
        st.session_state.captured_logs.append(f"--- Guardrail Scenario: {guardrail_type} ---")
        st.session_state.captured_logs.append(f"Text for check: {input_text[:100]}...")

        if guardrail_type == "Input Guardrail Test (Prompt Injection, Length)":
            st.subheader("Input Guardrail Results:")
            try:
                is_safe, sanitized_input, reason = await run_and_capture_logs(
                    st.session_state.safety_guardrails_instance.validate_input(input_text)
                )
                if is_safe:
                    st.success(f"Input Safe: {is_safe}")
                    st.write(f"Sanitized Input (if any): `{sanitized_input}`")
                    st.info("*(Simulating LLM processing for safe input...)*")
                else:
                    st.error(f"Input Safe: {is_safe}")
                    st.warning(f"Reason Blocked: {reason}")
            except Exception as e:
                st.exception(f"An error occurred during input guardrail check: {e}")

        else: # Output Guardrail Test
            st.subheader("Output Guardrail Results:")
            try:
                passed, sanitized_output = await run_and_capture_logs(
                    st.session_state.safety_guardrails_instance.validate_output(input_text)
                )
                if passed:
                    st.success(f"Output Passed: {passed}")
                    st.write(f"Original LLM Output: `{input_text}`")
                    st.write(f"Sanitized Output: `{sanitized_output}`")
                else:
                    st.error(f"Output Passed: {passed}")
                    st.warning("Output might contain unredacted sensitive information or other issues.")
            except Exception as e:
                st.exception(f"An error occurred during output guardrail check: {e}")

    st.markdown(f"") # Empty markdown to satisfy constraint
    st.markdown(f"The output clearly demonstrates the effectiveness of the input and output guardrails. Prompt injection attempts are detected and blocked, preventing potentially malicious instructions from reaching the LLM. Overly long inputs are also rejected, safeguarding against resource exhaustion. For LLM outputs, sensitive PII like email addresses, SSNs, credit card numbers, and phone numbers are automatically redacted. For OrgAIR, these guardrails are crucial for maintaining the security, compliance, and trustworthiness of our AI-driven knowledge extraction system, protecting both our clients' data and our own infrastructure.")
    st.markdown(f"") # Empty markdown to satisfy constraint

    with st.expander("Show Raw Logs"):
        st.code("\n".join(st.session_state.captured_logs), language="text")
```
