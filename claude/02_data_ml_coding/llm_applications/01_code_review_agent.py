"""
Problem: Smart Code Review Agent

Difficulty: Medium-Hard
Time: 60-90 minutes
AI Assistance: FULLY ALLOWED

Description:
Build an agentic system that performs automated code reviews. The agent should:
1. Analyze code for common issues (bugs, style, performance)
2. Provide specific, actionable feedback
3. Suggest improvements with code examples
4. Handle different programming languages
5. Be configurable for different review strictness levels

This problem is highly relevant to 8090's Software Factory, which uses
agentic systems for software development tasks.

Requirements:
-----------
Functional:
1. Accept code as input (string or file)
2. Identify issues across multiple categories:
   - Bugs and potential errors
   - Code style and readability
   - Performance issues
   - Security concerns
   - Best practices
3. Generate structured feedback with:
   - Issue description
   - Severity level (critical, high, medium, low)
   - Line numbers
   - Suggested fix with code example
4. Support multiple programming languages (at least Python, JavaScript)
5. Configurable review depth (quick, standard, thorough)

Non-Functional:
1. Response time < 30 seconds for typical code files
2. Cost-effective (minimize API calls)
3. Reliable error handling
4. Clear logging for debugging
5. Extensible architecture for adding new check types

Evaluation Criteria:
-------------------
1. Code Quality (30%)
   - Clean, modular architecture
   - Proper error handling
   - Comprehensive documentation
   - Type hints

2. Agent Design (25%)
   - Effective prompt engineering
   - Structured output parsing
   - Tool use (if applicable)
   - Error recovery

3. Functionality (25%)
   - Correctly identifies issues
   - Provides useful feedback
   - Handles edge cases
   - Multi-language support

4. Production Readiness (20%)
   - Configuration management
   - Logging and monitoring
   - Testing
   - Performance considerations

Example Usage:
-------------
```python
agent = CodeReviewAgent(
    model="gpt-4",
    review_depth="standard",
    languages=["python", "javascript"]
)

code = '''
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    return sum / len(numbers)
'''

review = agent.review_code(code, language="python")
print(review.summary)
for issue in review.issues:
    print(f"{issue.severity}: {issue.description}")
    print(f"Suggestion: {issue.suggestion}")
```

Expected Output:
---------------
Review Summary:
- 3 issues found (1 medium, 2 low)
- Overall code quality: Good
- Estimated fix time: 10 minutes

Issues:
1. [MEDIUM] Division by zero not handled (line 5)
   Suggestion: Add check for empty list before division
   ```python
   if not numbers:
       return 0
   return sum / len(numbers)
   ```

2. [LOW] Using built-in name 'sum' as variable (line 2)
   Suggestion: Rename variable to avoid shadowing built-in
   ```python
   total = 0
   ```

3. [LOW] Inefficient iteration pattern (line 3)
   Suggestion: Use direct iteration or built-in sum()
   ```python
   return sum(numbers) / len(numbers) if numbers else 0
   ```
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import os


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewDepth(Enum):
    """Review thoroughness levels"""
    QUICK = "quick"          # Fast, high-level review
    STANDARD = "standard"    # Balanced review
    THOROUGH = "thorough"    # Deep, comprehensive review


@dataclass
class CodeIssue:
    """Represents a single code issue found during review"""
    category: str              # e.g., "bug", "style", "performance"
    severity: Severity
    description: str
    line_number: Optional[int]
    suggestion: str
    code_example: Optional[str] = None


@dataclass
class CodeReview:
    """Complete code review results"""
    summary: str
    issues: List[CodeIssue]
    overall_quality: str       # e.g., "excellent", "good", "needs improvement"
    estimated_fix_time: str    # e.g., "5 minutes", "30 minutes"
    language: str


class CodeReviewAgent:
    """
    Agentic system for automated code review
    
    This agent uses LLMs to analyze code and provide structured feedback.
    It's designed to be extensible and production-ready.
    
    Parameters:
    -----------
    model : str
        LLM model to use (e.g., "gpt-4", "claude-3-sonnet")
    review_depth : ReviewDepth
        How thorough the review should be
    languages : List[str]
        Programming languages to support
    api_key : Optional[str]
        API key for LLM provider (if not in environment)
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        review_depth: ReviewDepth = ReviewDepth.STANDARD,
        languages: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ):
        # TODO: Initialize the agent
        pass
    
    def review_code(
        self,
        code: str,
        language: str,
        context: Optional[str] = None
    ) -> CodeReview:
        """
        Perform code review on the provided code
        
        Parameters:
        -----------
        code : str
            The code to review
        language : str
            Programming language of the code
        context : Optional[str]
            Additional context about the code (e.g., purpose, constraints)
            
        Returns:
        --------
        CodeReview
            Structured review results
        """
        # TODO: Implement code review logic
        pass
    
    def _build_review_prompt(
        self,
        code: str,
        language: str,
        context: Optional[str]
    ) -> str:
        """
        Build the prompt for the LLM
        
        This is a critical method - the quality of your prompt
        directly affects the quality of the review.
        """
        # TODO: Implement prompt engineering
        pass
    
    def _parse_review_response(self, response: str) -> CodeReview:
        """
        Parse the LLM response into structured CodeReview object
        
        Handle cases where the LLM doesn't follow the expected format.
        """
        # TODO: Implement response parsing
        pass
    
    def _validate_code(self, code: str, language: str) -> None:
        """
        Validate input code
        
        Check for:
        - Empty code
        - Supported language
        - Reasonable code length
        """
        # TODO: Implement validation
        pass


def test_code_review_agent():
    """Test the CodeReviewAgent implementation"""
    
    # Test case 1: Python code with issues
    print("Test 1: Python code with common issues")
    print("-" * 70)
    
    code = """
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    return sum / len(numbers)
"""
    
    agent = CodeReviewAgent(
        model="gpt-4",
        review_depth=ReviewDepth.STANDARD,
        languages=["python"]
    )
    
    review = agent.review_code(code, language="python")
    
    print(f"Summary: {review.summary}")
    print(f"Overall Quality: {review.overall_quality}")
    print(f"Issues Found: {len(review.issues)}")
    
    for i, issue in enumerate(review.issues, 1):
        print(f"\n{i}. [{issue.severity.value.upper()}] {issue.description}")
        if issue.line_number:
            print(f"   Line: {issue.line_number}")
        print(f"   Suggestion: {issue.suggestion}")
        if issue.code_example:
            print(f"   Example:\n{issue.code_example}")
    
    # Test case 2: JavaScript code
    print("\n\nTest 2: JavaScript code")
    print("-" * 70)
    
    js_code = """
function fetchData(url) {
    var result = fetch(url);
    return result.json();
}
"""
    
    review = agent.review_code(js_code, language="javascript")
    print(f"Issues Found: {len(review.issues)}")
    
    print("\n✅ Tests completed!")


if __name__ == "__main__":
    # Note: You'll need to set OPENAI_API_KEY or ANTHROPIC_API_KEY
    # in your environment to run this
    
    test_code_review_agent()

