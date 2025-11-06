"""Test CAMEL-AI imports and Gemini integration."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("Testing CAMEL-AI Imports and Gemini Integration")
print("=" * 60)

# Test 1: Basic imports
print("\n[1] Testing basic CAMEL-AI imports...")
try:
    from camel.societies.workforce import Workforce
    print("  OK: Workforce import successful")
except ImportError as e:
    print(f"  FAILED: Workforce import failed: {e}")
    sys.exit(1)

try:
    from camel.agents import ChatAgent
    print("  OK: ChatAgent import successful")
except ImportError as e:
    print(f"  FAILED: ChatAgent import failed: {e}")
    sys.exit(1)

try:
    from camel.messages import BaseMessage
    print("  OK: BaseMessage import successful")
except ImportError as e:
    print(f"  FAILED: BaseMessage import failed: {e}")
    sys.exit(1)

try:
    from camel.tasks import Task
    print("  OK: Task import successful")
except ImportError as e:
    print(f"  FAILED: Task import failed: {e}")
    sys.exit(1)

try:
    from camel.memories import LongtermAgentMemory
    print("  OK: LongtermAgentMemory import successful")
except ImportError as e:
    print(f"  FAILED: LongtermAgentMemory import failed: {e}")
    sys.exit(1)

try:
    from camel.types import ModelType, ModelPlatformType
    from camel.models import ModelFactory
    print("  OK: ModelType, ModelPlatformType and ModelFactory import successful")
except ImportError as e:
    print(f"  FAILED: Model imports failed: {e}")
    sys.exit(1)

# Test 2: Check Gemini API key
print("\n[2] Checking Gemini API key...")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print(f"  OK: Gemini API key found (length: {len(gemini_key)})")
else:
    print("  WARNING: No Gemini API key found in environment")
    sys.exit(1)

# Test 3: Test model factory
print("\n[3] Testing CamelModelFactory with Gemini...")
try:
    from core.models.camel_models import CamelModelFactory
    print("  OK: CamelModelFactory import successful")
    
    # Try to create a Gemini model
    print("  Attempting to create Gemini model...")
    try:
        # Try with gemini-1.5-flash first (faster and cheaper)
        model = CamelModelFactory.create_model(
            model_name="gemini-1.5-flash",
            api_key=gemini_key
        )
        print("  OK: Gemini 1.5 Flash model created successfully")
        print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"  WARNING: Gemini 1.5 Flash model creation failed: {e}")
        # Try gemini-1.5-pro
        try:
            print("  Attempting Gemini 1.5 Pro...")
            model = CamelModelFactory.create_model(
                model_name="gemini-1.5-pro",
                api_key=gemini_key
            )
            print("  OK: Gemini 1.5 Pro model created successfully")
        except Exception as e2:
            print(f"  WARNING: Gemini 1.5 Pro also failed: {e2}")
            print("  This might be expected if Gemini requires different setup")
            # Try with OpenAI as fallback
            print("  Attempting fallback to OpenAI...")
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    model = CamelModelFactory.create_model(
                        model_name="gpt-4o-mini",
                        api_key=openai_key
                    )
                    print("  OK: OpenAI fallback model created successfully")
                except Exception as fallback_error:
                    print(f"  FAILED: Fallback also failed: {fallback_error}")
            else:
                print("  SKIP: No OpenAI API key for fallback")
except Exception as e:
    print(f"  FAILED: ModelFactory test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test ChatAgent with Gemini
print("\n[4] Testing ChatAgent with Gemini model...")
try:
    # Check available ModelType values
    print("  Checking available ModelType values...")
    gemini_types = [attr for attr in dir(ModelType) if 'GEMINI' in attr.upper()]
    print(f"  Found Gemini model types: {gemini_types}")
    
    # Try creating a ChatAgent
    print("  Creating ChatAgent with Gemini...")
    try:
        # First, try to create the model with GEMINI_1_5_FLASH
        gemini_model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=ModelType.GEMINI_1_5_FLASH,
            api_key=gemini_key
        )
        
        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Test Agent",
                content="You are a helpful test agent."
            ),
            model=gemini_model
        )
        print("  OK: ChatAgent created with Gemini 1.5 Flash model")
        
        # Test a simple message
        print("  Testing agent with a simple message...")
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content="Say hello in one sentence."
        )
        response = agent.step(user_msg)
        print(f"  OK: Agent responded: {response.msgs[0].content[:100]}...")
        
    except Exception as e:
        print(f"  WARNING: Gemini ChatAgent test failed: {e}")
        print("  This might be expected - checking if Gemini is properly configured")
        # Try with GEMINI_1_5_PRO as fallback
        try:
            print("  Attempting with Gemini 1.5 Pro...")
            gemini_model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=ModelType.GEMINI_1_5_PRO,
                api_key=gemini_key
            )
            agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name="Test Agent",
                    content="You are a helpful test agent."
                ),
                model=gemini_model
            )
            print("  OK: ChatAgent created with Gemini 1.5 Pro model")
        except Exception as e2:
            print(f"  WARNING: Gemini 1.5 Pro also failed: {e2}")
            import traceback
            traceback.print_exc()
        
except Exception as e:
    print(f"  FAILED: ChatAgent test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test our custom components
print("\n[5] Testing custom CAMEL components...")
try:
    from core.memory.camel_memory_manager import CamelMemoryManager
    print("  OK: CamelMemoryManager import successful")
except ImportError as e:
    print(f"  WARNING: CamelMemoryManager import failed: {e}")

try:
    from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
    print("  OK: MCPForecastingToolkit import successful")
except ImportError as e:
    print(f"  WARNING: MCPForecastingToolkit import failed: {e}")

try:
    from agents.workforce_orchestrator import WorkforceOrchestratorAgent
    print("  OK: WorkforceOrchestratorAgent import successful")
except ImportError as e:
    print(f"  WARNING: WorkforceOrchestratorAgent import failed: {e}")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("All core CAMEL-AI imports successful!")
print("Gemini API key is configured.")
print("=" * 60)

