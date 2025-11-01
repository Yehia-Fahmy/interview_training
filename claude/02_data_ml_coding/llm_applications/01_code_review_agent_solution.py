"""
Solution: Smart Code Review Agent

This is a production-quality implementation demonstrating:
- Effective prompt engineering
- Structured output parsing
- Error handling and validation
- Logging and monitoring
- Extensible architecture
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewDepth(Enum):
    """Review thoroughness levels"""
    QUICK = "quick"
    STANDARD = "standard"
    THOROUGH = "thorough"


@dataclass
class CodeIssue:
    """Represents a single code issue"""
    category: str
    severity: Severity
    description: str
    line_number: Optional[int]
    suggestion: str
    code_example: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


@dataclass
class CodeReview:
    """Complete code review results"""
    summary: str
    issues: List[CodeIssue]
    overall_quality: str
    estimated_fix_time: str
    language: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'summary': self.summary,
            'issues': [issue.to_dict() for issue in self.issues],
            'overall_quality': self.overall_quality,
            'estimated_fix_time': self.estimated_fix_time,
            'language': self.language,
            'timestamp': self.timestamp
        }


class CodeReviewAgent:
    """
    Agentic system for automated code review
    
    Example:
    --------
    >>> agent = CodeReviewAgent(model="gpt-4")
    >>> review = agent.review_code(code, language="python")
    >>> print(review.summary)
    """
    
    # Maximum code length to review (tokens)
    MAX_CODE_LENGTH = 4000
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "java",
        "go", "rust", "cpp", "c", "ruby", "php"
    ]
    
    def __init__(
        self,
        model: str = "gpt-4",
        review_depth: ReviewDepth = ReviewDepth.STANDARD,
        languages: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.review_depth = review_depth
        self.languages = languages or self.SUPPORTED_LANGUAGES
        
        # Initialize LLM client
        self._init_llm_client(api_key)
        
        logger.info(
            f"Initialized CodeReviewAgent with model={model}, "
            f"depth={review_depth.value}"
        )
    
    def _init_llm_client(self, api_key: Optional[str]):
        """Initialize the LLM client (OpenAI or Anthropic)"""
        try:
            if "gpt" in self.model.lower():
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.provider = "openai"
            elif "claude" in self.model.lower():
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
                self.provider = "anthropic"
            else:
                raise ValueError(f"Unsupported model: {self.model}")
        except ImportError as e:
            raise ImportError(
                f"Required library not installed. "
                f"Install with: pip install {'openai' if 'gpt' in self.model else 'anthropic'}"
            )
    
    def review_code(
        self,
        code: str,
        language: str,
        context: Optional[str] = None
    ) -> CodeReview:
        """
        Perform code review
        
        Parameters:
        -----------
        code : str
            Code to review
        language : str
            Programming language
        context : Optional[str]
            Additional context
            
        Returns:
        --------
        CodeReview
            Structured review results
        """
        logger.info(f"Starting code review for {language} code ({len(code)} chars)")
        
        # Validate inputs
        self._validate_code(code, language)
        
        try:
            # Build prompt
            prompt = self._build_review_prompt(code, language, context)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            review = self._parse_review_response(response, language)
            
            logger.info(f"Review completed: {len(review.issues)} issues found")
            return review
            
        except Exception as e:
            logger.error(f"Error during code review: {e}", exc_info=True)
            raise
    
    def _build_review_prompt(
        self,
        code: str,
        language: str,
        context: Optional[str]
    ) -> str:
        """
        Build the review prompt
        
        Key prompt engineering techniques:
        1. Clear role definition
        2. Structured output format
        3. Specific examples
        4. Severity guidelines
        5. Context incorporation
        """
        
        # Depth-specific instructions
        depth_instructions = {
            ReviewDepth.QUICK: "Focus on critical bugs and major issues only.",
            ReviewDepth.STANDARD: "Review for bugs, style, and performance issues.",
            ReviewDepth.THOROUGH: "Perform comprehensive review including security, maintainability, and best practices."
        }
        
        prompt = f"""You are an expert code reviewer specializing in {language}.

Review the following code and provide structured feedback.

REVIEW DEPTH: {self.review_depth.value}
{depth_instructions[self.review_depth]}

CODE TO REVIEW:
```{language}
{code}
```

{f"CONTEXT: {context}" if context else ""}

Provide your review in the following JSON format:
{{
    "summary": "Brief overview of the code quality and main findings",
    "overall_quality": "excellent|good|fair|needs_improvement|poor",
    "estimated_fix_time": "Estimated time to fix all issues (e.g., '10 minutes', '1 hour')",
    "issues": [
        {{
            "category": "bug|style|performance|security|best_practice",
            "severity": "critical|high|medium|low|info",
            "description": "Clear description of the issue",
            "line_number": <line number or null>,
            "suggestion": "Specific, actionable suggestion",
            "code_example": "Fixed code example (optional)"
        }}
    ]
}}

SEVERITY GUIDELINES:
- CRITICAL: Security vulnerabilities, data loss, crashes
- HIGH: Bugs that affect functionality
- MEDIUM: Performance issues, poor practices
- LOW: Style issues, minor improvements
- INFO: Suggestions and tips

IMPORTANT:
1. Be specific and actionable
2. Include line numbers when possible
3. Provide code examples for fixes
4. Focus on the most impactful issues first
5. Return ONLY valid JSON, no additional text

Your response:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert code reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent output
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_review_response(self, response: str, language: str) -> CodeReview:
        """
        Parse LLM response into CodeReview object
        
        Handles cases where LLM doesn't return perfect JSON.
        """
        try:
            # Extract JSON from response (in case LLM added extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Parse issues
            issues = []
            for issue_data in data.get('issues', []):
                try:
                    issue = CodeIssue(
                        category=issue_data['category'],
                        severity=Severity(issue_data['severity']),
                        description=issue_data['description'],
                        line_number=issue_data.get('line_number'),
                        suggestion=issue_data['suggestion'],
                        code_example=issue_data.get('code_example')
                    )
                    issues.append(issue)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed issue: {e}")
                    continue
            
            # Create CodeReview object
            review = CodeReview(
                summary=data.get('summary', 'No summary provided'),
                issues=issues,
                overall_quality=data.get('overall_quality', 'unknown'),
                estimated_fix_time=data.get('estimated_fix_time', 'unknown'),
                language=language
            )
            
            return review
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            
            # Fallback: create a basic review
            return CodeReview(
                summary="Failed to parse review response",
                issues=[],
                overall_quality="unknown",
                estimated_fix_time="unknown",
                language=language
            )
        
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    def _validate_code(self, code: str, language: str) -> None:
        """Validate input code"""
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        if language.lower() not in [lang.lower() for lang in self.languages]:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {', '.join(self.languages)}"
            )
        
        if len(code) > self.MAX_CODE_LENGTH * 4:  # Rough token estimate
            raise ValueError(
                f"Code too long ({len(code)} chars). "
                f"Maximum: ~{self.MAX_CODE_LENGTH * 4} chars"
            )


def test_code_review_agent():
    """Comprehensive test suite"""
    
    print("="*70)
    print("TESTING CODE REVIEW AGENT")
    print("="*70)
    
    # Test 1: Python code with multiple issues
    print("\nTest 1: Python code with common issues")
    print("-" * 70)
    
    python_code = """
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    return sum / len(numbers)

def fetch_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""
    
    try:
        agent = CodeReviewAgent(
            model="gpt-4",
            review_depth=ReviewDepth.STANDARD
        )
        
        review = agent.review_code(python_code, language="python")
        
        print(f"\nSummary: {review.summary}")
        print(f"Overall Quality: {review.overall_quality}")
        print(f"Estimated Fix Time: {review.estimated_fix_time}")
        print(f"Issues Found: {len(review.issues)}\n")
        
        for i, issue in enumerate(review.issues, 1):
            print(f"{i}. [{issue.severity.value.upper()}] {issue.category}")
            print(f"   {issue.description}")
            if issue.line_number:
                print(f"   Line: {issue.line_number}")
            print(f"   Suggestion: {issue.suggestion}")
            if issue.code_example:
                print(f"   Example:\n{issue.code_example}")
            print()
        
        # Save review to file
        with open('review_output.json', 'w') as f:
            json.dump(review.to_dict(), f, indent=2)
        print("✓ Review saved to 'review_output.json'")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return
    
    # Test 2: Different review depths
    print("\n" + "="*70)
    print("Test 2: Different review depths")
    print("-" * 70)
    
    simple_code = "def add(a, b):\n    return a + b"
    
    for depth in [ReviewDepth.QUICK, ReviewDepth.STANDARD, ReviewDepth.THOROUGH]:
        agent = CodeReviewAgent(review_depth=depth)
        review = agent.review_code(simple_code, language="python")
        print(f"{depth.value}: {len(review.issues)} issues found")
    
    print("\n✅ All tests completed!")


def demo_with_real_code():
    """Demo with a more realistic code example"""
    
    print("\n" + "="*70)
    print("DEMO: Reviewing Real-World Code")
    print("="*70 + "\n")
    
    code = """
import requests

class UserService:
    def __init__(self, api_url):
        self.api_url = api_url
    
    def get_user(self, user_id):
        response = requests.get(f"{self.api_url}/users/{user_id}")
        return response.json()
    
    def create_user(self, user_data):
        response = requests.post(
            f"{self.api_url}/users",
            json=user_data
        )
        return response.json()
    
    def update_user(self, user_id, user_data):
        url = self.api_url + "/users/" + str(user_id)
        r = requests.put(url, json=user_data)
        return r.json()
"""
    
    agent = CodeReviewAgent(
        model="gpt-4",
        review_depth=ReviewDepth.THOROUGH
    )
    
    review = agent.review_code(
        code,
        language="python",
        context="This is a service class for user management in a web application"
    )
    
    print(f"Summary: {review.summary}\n")
    print(f"Quality: {review.overall_quality}")
    print(f"Fix Time: {review.estimated_fix_time}\n")
    print(f"Found {len(review.issues)} issues:\n")
    
    for issue in review.issues:
        print(f"[{issue.severity.value.upper()}] {issue.description}")
        print(f"→ {issue.suggestion}\n")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No API key found in environment")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run tests")
        print("\nExample:")
        print("export OPENAI_API_KEY='your-key-here'")
    else:
        test_code_review_agent()
        demo_with_real_code()
    
    # Print architecture notes
    print("\n" + "="*70)
    print("ARCHITECTURE NOTES")
    print("="*70)
    print("""
Key Design Decisions:
--------------------
1. Structured Output: Use JSON for reliable parsing
2. Enum Types: Type-safe severity and depth levels
3. Dataclasses: Clean data modeling
4. Logging: Comprehensive logging for debugging
5. Error Handling: Graceful degradation
6. Validation: Input validation prevents issues
7. Extensibility: Easy to add new languages/checks

Production Improvements:
-----------------------
1. Caching: Cache reviews for identical code
2. Rate Limiting: Respect API rate limits
3. Async: Use async for parallel reviews
4. Retries: Implement retry logic for API failures
5. Metrics: Track review quality and performance
6. Custom Rules: Allow user-defined review rules
7. Integration: GitHub/GitLab webhook integration

Prompt Engineering Tips:
-----------------------
1. Clear role definition
2. Structured output format (JSON)
3. Specific examples and guidelines
4. Temperature tuning (lower = more consistent)
5. Token limits for cost control
6. Few-shot examples for complex tasks
    """)

