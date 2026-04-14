# NOVA AI Platform - Coding Patterns

## Code Standards
- **Language**: Python 3.10+
- **Type hints**: Required on all function signatures
- **Docstrings**: Google-style for all public functions/classes
- **Line length**: 120 max
- **Imports**: stdlib → third-party → local, alphabetized within groups
- **Naming**: snake_case for functions/variables, PascalCase for classes

## Error Handling
```python
# Use custom exceptions
class NOVAError(Exception): pass
class IntentClassificationError(NOVAError): pass
class ToolExecutionError(NOVAError): pass
class EscalationError(NOVAError): pass

# Always log errors
import logging
logger = logging.getLogger(__name__)

try:
    result = tool.execute(**params)
except ToolExecutionError as e:
    logger.error(f"Tool execution failed: {e}", extra={"tool": tool.name})
    raise
```

## LLM Calls Pattern
```python
from langchain_openai import ChatOpenAI
from config.settings import Config

llm = ChatOpenAI(
    api_key=Config.llm.api_key,
    base_url=Config.llm.base_url,
    model=Config.llm.model_name,
    temperature=Config.llm.temperature,
    max_tokens=Config.llm.max_tokens,
)
```

## Tool Pattern
```python
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Validated input schema."""
    query: str

class ToolOutput(BaseModel):
    """Structured output schema."""
    success: bool
    data: dict
    error: str | None = None

async def tool_function(input: ToolInput) -> ToolOutput:
    """Execute tool with audit logging."""
    # 1. Validate input
    # 2. Execute logic
    # 3. Log to audit trail
    # 4. Return structured output
```

## Testing Pattern
```python
import pytest

class TestComponent:
    """Test suite for component."""

    @pytest.fixture
    def setup(self):
        """Shared test fixtures."""
        return Component()

    def test_happy_path(self, setup):
        """Test normal operation."""
        result = setup.process(valid_input)
        assert result.success

    def test_error_case(self, setup):
        """Test error handling."""
        with pytest.raises(ExpectedError):
            setup.process(invalid_input)

    @pytest.mark.asyncio
    async def test_async_operation(self, setup):
        """Test async operations."""
        result = await setup.async_process(input)
        assert result.success
```

## Brand Voice Guidelines
- Friendly, warm, enthusiastic tone
- Use emojis sparingly but effectively (💕, ✨, 🎉, 📦)
- Address customer by name when available
- Proactive suggestions and follow-up questions
- Never robotic or overly formal
- Acknowledge emotions before solving problems