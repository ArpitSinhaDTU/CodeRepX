import os
import sys
import io
import re
import glob
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import google.generativeai as genai

# ====================== CONFIGURATION ======================
INPUT_DIR = r"C:\Users\Acer\OneDrive\Desktop\CodeRepX-main\python_programs"
OUTPUT_DIR = r"C:\Users\Acer\OneDrive\Desktop\CodeRepX-main\fixed_programs"
TESTCASE_DIR = r"C:\Users\Acer\OneDrive\Desktop\CodeRepX-main\python_testcases"
# ===========================================================

# Load environment variables
load_dotenv()

# Initialize models
groq_fixer = ChatGroq(
    temperature=0.1,
    model_name="deepseek-r1-distill-llama-70b",
    groq_api_key="YOUR_GROQ_API_KEY"
)

groq_verifier = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    groq_api_key="YOUR_GROQ_API_KEY"
)

groq_test_validator = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    groq_api_key="YOUR_GROQ_API_KEY"
)

groq_llama_fixer = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    groq_api_key="YOUR_GROQ_API_KEY"
)

genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_classifier = genai.GenerativeModel('models/gemini-2.0-flash')
gemini_fixer = genai.GenerativeModel('models/gemini-2.0-flash')

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TESTCASE_DIR, exist_ok=True)

# Updated defect classes
DEFECT_CLASSES = """1. Incorrect assignment operator: Example - Using '=' instead of '==' in condition
2. Incorrect variable: Example - Using wrong variable name
3. Incorrect comparison operator: Example - Wrong comparison operator
4. Missing condition: Example - Missing edge case check
5. Missing/added +1: Example - Off-by-one error
6. Variable swap: Example - Swapped variables
7. Incorrect array slice: Example - Wrong slicing bounds
8. Variable prepend: Example - Missing variable prefix
9. Incorrect data structure constant: Example - Wrong initialization
10. Incorrect method called: Example - Wrong method name
11. Incorrect field dereference: Example - Accessing wrong attribute
12. Missing arithmetic expression: Example - Incomplete calculation
13. Missing function call: Example - Forgetting to call function
14. Missing line: Example - Critical line omitted"""

# Enhanced code extraction function
def extract_pure_python_code(response_text):
    """
    Extract pure Python code from model response using multiple strategies:
    1. Look for markdown code blocks
    2. Look for code after specific markers
    3. Remove any non-Python content
    """
    # Strategy 1: Extract markdown code blocks
    code_match = re.search(r'```python(.*?)```', response_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Strategy 2: Extract after specific markers
    for marker in ["Act:", "Corrected Code:", "Fixed Code:", "```"]:
        if marker in response_text:
            parts = response_text.split(marker)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                # If we have more than 5 lines, assume it's code
                if candidate.count('\n') > 4:
                    return candidate
    
    # Strategy 3: Remove all lines that don't look like Python code
    lines = response_text.split('\n')
    python_lines = []
    code_started = False
    
    for line in lines:
        # Check if line looks like Python code
        if re.match(r'^\s*(def |class |import |from |\w+ = |if |for |while |print)', line):
            code_started = True
        if code_started and not line.startswith(('Reasoning:', 'Explanation:', 'Thought:')):
            python_lines.append(line)
    
    result = '\n'.join(python_lines).strip()
    
    # Final cleanup: Remove any remaining non-code prefixes
    for prefix in ["Here's the fixed code:", "Fixed version:", "Corrected code:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    
    return result

# Agent 1: Gemini Classifier (with original error context)
def gemini_classifier_agent(code, original_error):
    prompt = f"""Analyze this code step-by-step to classify the defect:
1. Identify possible defects from: {DEFECT_CLASSES}
2. Compare each possibility to the code and error
3. Explain your reasoning. Choose the error only from the 14 defect classes.
4. Final answer must be: "<number>. <defect name> + <description on how to solve the problem with example>"

Original Error:
{original_error}

Code:
{code}

Reasoning Steps:"""
    response = gemini_classifier.generate_content(prompt)
    return response.text

# Agent 2: DeepSeek Fixer
def deepseek_fixer_agent(code, defect_class, feedback, original_error, current_error):
    prompt = f"""Fix this Python code:
Defect: {defect_class}
Original Error: {original_error}
Current Error: {current_error}
Previous Feedback: {feedback}

Code:
{code}

IMPORTANT:
- Return ONLY the fixed Python code
- Do NOT include any explanations, comments, or markdown
- Do NOT wrap code in ```python blocks
- Maintain the original formatting, comments, and structure as Approximately everychange in one liner.This is important to ensure the code runs correctly.
- Do NOT add any text before or after the code
- Ensure the code is complete and executable

Fixed Code:"""
    
    response = groq_fixer.invoke(prompt)
    fixed_code = extract_pure_python_code(response.content)
    
    # Ensure we have valid Python code
    if not fixed_code.strip() or fixed_code.count('\n') < 3:
        fixed_code = code  # Fallback to original if bad output
    
    return fixed_code

# Agent 3: Gemini Fixer
def gemini_fixer_agent(code, defect_class, feedback, original_error, current_error):
    prompt = f"""Fix this Python code:
Defect: {defect_class}
Original Error: {original_error}
Current Error: {current_error}
Previous Feedback: {feedback}

Code:
{code}

IMPORTANT:
- Return ONLY the fixed Python code
- Do NOT include any explanations, comments, or markdown
- Maintain the original formatting, comments, and structure as Approximately everychange in one liner.This is important to ensure the code runs correctly.
- Do NOT wrap code in ```python blocks
- Do NOT add any text before or after the code
- Ensure the code is complete and executable

Fixed Code:"""
    
    response = gemini_fixer.generate_content(prompt)
    fixed_code = extract_pure_python_code(response.text)
    
    # Ensure we have valid Python code
    if not fixed_code.strip() or fixed_code.count('\n') < 3:
        fixed_code = code  # Fallback to original if bad output
    
    return fixed_code

# Agent 4: Llama Fixer
def llama_fixer_agent(code, defect_class, feedback, original_error, current_error):
    prompt = f"""Fix this Python code:
Defect: {defect_class}
Original Error: {original_error}
Current Error: {current_error}
Previous Feedback: {feedback}

Code:
{code}

IMPORTANT:
- Return ONLY the fixed Python code
- Do NOT include any explanations, comments, or markdown
- Do NOT wrap code in ```python blocks
- Maintain the original formatting, comments, and structure as Approximately everychange in one liner.This is important to ensure the code runs correctly.
- Do NOT add any text before or after the code
- Ensure the code is complete and executable

Fixed Code:"""
    
    response = groq_llama_fixer.invoke(prompt)
    fixed_code = extract_pure_python_code(response.content)
    
    # Ensure we have valid Python code
    if not fixed_code.strip() or fixed_code.count('\n') < 3:
        fixed_code = code  # Fallback to original if bad output
    
    return fixed_code

# Generate fixes using multiple models
def generate_fixes_with_multiple_models(code, defect_class, feedback, original_error, current_error):
    """Generate multiple fixed versions using different models"""
    models = [
        ("DeepSeek", lambda: deepseek_fixer_agent(code, defect_class, feedback, original_error, current_error)),
        ("Gemini", lambda: gemini_fixer_agent(code, defect_class, feedback, original_error, current_error)),
        ("Llama", lambda: llama_fixer_agent(code, defect_class, feedback, original_error, current_error))
    ]
    
    fixes = {}
    for name, fixer_fn in models:
        try:
            fixed_code = fixer_fn()
            if fixed_code.strip():
                fixes[name] = fixed_code
                print(f"\nGenerated {name} Fix:")
                print("-"*60)
                print(fixed_code[:500] + ("..." if len(fixed_code) > 500 else ""))
                print("-"*60)
            else:
                print(f"Empty fix from {name}")
        except Exception as e:
            print(f"Error in {name} fixer: {str(e)}")
    
    return fixes

# Enhanced Code Execution Tool
def execute_code(code):
    """Execute Python code and capture detailed error information"""
    output = io.StringIO()
    error = io.StringIO()
    
    try:
        # Create a safe environment with minimal globals
        safe_globals = {'__builtins__': None}
        with redirect_stdout(output), redirect_stderr(error):
            exec(code, safe_globals)
        stderr_content = error.getvalue()
        return {
            "stdout": output.getvalue(),
            "stderr": stderr_content,
            "error_type": "NONE",
            "error_message": stderr_content,
            "success": stderr_content == ""
        }
    except Exception as e:
        # Capture detailed error information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        
        return {
            "stdout": output.getvalue(),
            "stderr": tb_text,
            "error_type": exc_type.__name__ if exc_type else "Exception",
            "error_message": str(e),
            "success": False
        }

# Test Case Extraction Tool
def get_test_case(program_name):
    """Find and return the test case for a given program"""
    # Try multiple patterns to find the test case
    patterns = [
        f"test_{program_name}.py",
        f"test_{program_name}_test.py",
        f"{program_name}_test.py",
        f"test_{program_name.replace('_', '')}.py"
    ]
    
    for pattern in patterns:
        test_path = os.path.join(TESTCASE_DIR, pattern)
        if os.path.exists(test_path):
            try:
                with open(test_path, "r") as f:
                    return f.read(), os.path.basename(test_path)
            except Exception as e:
                print(f"Error reading test case {pattern}: {str(e)}")
                continue
    
    # If no exact match, try case-insensitive search
    all_test_files = glob.glob(os.path.join(TESTCASE_DIR, "test_*.py"))
    for test_file in all_test_files:
        base_name = os.path.basename(test_file)
        if program_name.lower() in base_name.lower():
            try:
                with open(test_file, "r") as f:
                    return f.read(), os.path.basename(test_file)
            except Exception as e:
                print(f"Error reading test case {test_file}: {str(e)}")
    
    return None, None

# Agent 5: Test Case Validator (with detailed error context)
def test_case_validator_agent(original_code, fixed_code, defect_class, test_case, execution_result, test_case_name, original_error):
    # Create detailed execution status message
    if execution_result['success']:
        exec_status = "SUCCESS"
    else:
        error_details = (
            f"ERROR TYPE: {execution_result['error_type']}\n"
            f"ERROR MESSAGE: {execution_result['error_message']}\n"
            f"FULL TRACEBACK:\n{execution_result['stderr'][:500]}"
        )
        exec_status = f"EXECUTION FAILED\n{error_details}"
    
    prompt = f"""Validate this fix for '{defect_class}' in 1-2 sentences:
- Original Error: {original_error}
- Execution: {exec_status}
- Test Case: {test_case_name}

RULES:
1. REJECT if execution failed
2. Only APPROVE if fix is correct and test passes
3. Feedback MUST be concise (1-2 sentences)
4. Provide SPECIFIC code change instructions based on error
5. Compare to original error: {original_error}

Return ONLY:
VERDICT: [APPROVED/REJECTED]
FEEDBACK: <concise instructions>"""
    
    print("\n" + "="*80)
    print("TEST CASE VALIDATION PROCESS")
    print("="*80)
    print(f"Using test case: {test_case_name}")
    print(f"\nTarget Defect: {defect_class}")
    print(f"\nOriginal Error: {original_error}")
    print(f"\nExecution Result: {exec_status}")
    
    response = groq_test_validator.invoke(prompt)
    content = response.content
    
    # Enforce rejection if execution failed
    if not execution_result['success']:
        if "VERDICT: APPROVED" in content:
            content = "VERDICT: REJECTED\nFEEDBACK: Execution failed - fix the runtime error"
        elif "VERDICT:" not in content:
            content = "VERDICT: REJECTED\nFEEDBACK: Execution failed - " + execution_result['error_message'][:150]
    
    print(f"\nVerification Result:\n{'-'*40}\n{content}\n{'-'*40}")
    
    return content

# Agent 6: Llama Verifier (with detailed error context)
def llama_verifier_agent(original, fixed, defect_class, execution_result, test_case, test_case_name, original_error):
    # Create detailed execution status message
    if execution_result['success']:
        exec_status = "SUCCESS"
    else:
        error_details = (
            f"ERROR TYPE: {execution_result['error_type']}\n"
            f"ERROR MESSAGE: {execution_result['error_message']}\n"
            f"FULL TRACEBACK:\n{execution_result['stderr'][:500]}"
        )
        exec_status = f"EXECUTION FAILED\n{error_details}"
    
    if test_case:
        return test_case_validator_agent(original, fixed, defect_class, test_case, execution_result, test_case_name, original_error)
    
    # Fallback verification (with error context)
    prompt = f"""Verify this fix for '{defect_class}' in 1-2 sentences:
- Original Error: {original_error}
- Execution: {exec_status}

RULES:
1. REJECT if execution failed
2. Only APPROVE if fix is correct
3. Feedback MUST be concise (1-2 sentences)
4. Provide SPECIFIC code change instructions based on error
5. Compare to original error: {original_error}

Return ONLY:
VERDICT: [APPROVED/REJECTED]
FEEDBACK: <concise instructions>"""
    
    print("\n" + "="*80)
    print("VERIFICATION PROCESS (NO TEST CASE)")
    print("="*80)
    print(f"Target Defect: {defect_class}")
    print(f"\nOriginal Error: {original_error}")
    print(f"\nExecution Result: {exec_status}")
    
    response = groq_verifier.invoke(prompt)
    content = response.content
    
    # Enforce rejection if execution failed
    if not execution_result['success']:
        if "VERDICT: APPROVED" in content:
            content = "VERDICT: REJECTED\nFEEDBACK: Execution failed - fix the runtime error"
        elif "VERDICT:" not in content:
            content = "VERDICT: REJECTED\nFEEDBACK: Execution failed - " + execution_result['error_message'][:150]
    
    print(f"\nVerification Result:\n{'-'*40}\n{content}\n{'-'*40}")
    
    return content

# Modified repair workflow with ensemble fixing
def repair_workflow(original_code, filename, max_attempts=5):
    attempts = 0
    current_code = original_code
    defect_class = None
    feedback = "Initial attempt"
    program_name = os.path.splitext(filename)[0]
    test_case_content, test_case_name = get_test_case(program_name)
    
    print(f"\n{'='*80}")
    print(f"STARTING REPAIR PROCESS FOR: {filename}")
    print("="*80)
    if test_case_name:
        print(f"Test case found: {test_case_name}")
    else:
        print("Test case not found")
    
    # Execute original code to capture initial error
    print("\n[INITIAL EXECUTION] Running original code...")
    original_execution = execute_code(original_code)
    original_error = (
        f"Error Type: {original_execution['error_type']}\n"
        f"Error Message: {original_execution['error_message']}\n"
        f"Traceback:\n{original_execution['stderr'][:500]}"
    ) if not original_execution['success'] else "No error - code executed successfully"
    
    print(f"Original Code Execution: {'SUCCESS' if original_execution['success'] else 'FAILED'}")
    if not original_execution['success']:
        print(f"Original Error:\n{original_error}")
    
    while attempts < max_attempts:
        try:
            print(f"\n{'='*60}")
            print(f"ATTEMPT {attempts+1}/{max_attempts}")
            print("="*60)
            
            # Classify defect with Gemini (include original error)
            if attempts == 0:
                print("\n[AGENT 1] Classifying defect with original error context...")
                defect_class = gemini_classifier_agent(current_code, original_error)
                print(f"Defect Classification: {defect_class}")
            
            # Generate multiple fixes with different models
            print("\n[AGENT 2-4] Generating fixes with multiple models...")
            current_execution = execute_code(current_code)
            current_error = (
                f"Error Type: {current_execution['error_type']}\n"
                f"Error Message: {current_execution['error_message']}"
            ) if not current_execution['success'] else "No error"
            
            fixes = generate_fixes_with_multiple_models(
                current_code,
                defect_class,
                feedback,
                original_error,
                current_error
            )
            
            # Evaluate each fix
            best_fix = None
            best_feedback = ""
            model_ranking = []
            
            for model_name, fixed_code in fixes.items():
                print(f"\n[EVALUATION] Testing {model_name}'s fix...")
                
                # Execute the fixed code
                execution_result = execute_code(fixed_code)
                exec_status = 'SUCCESS' if execution_result['success'] else 'ERROR'
                print(f"Execution Status: {exec_status}")
                
                if execution_result['success']:
                    print(f"{model_name} fix executed successfully!")
                    model_ranking.append(model_name)
                
                # Verify with verifier agent
                verification = llama_verifier_agent(
                    original_code,
                    fixed_code,
                    defect_class,
                    execution_result,
                    test_case_content,
                    test_case_name,
                    original_error
                )
                
                # Extract verdict and feedback
                verdict_match = re.search(r'VERDICT:\s*(\w+)', verification, re.IGNORECASE)
                feedback_match = re.search(r'FEEDBACK:\s*(.+)', verification, re.DOTALL)
                
                verdict = verdict_match.group(1).upper() if verdict_match else "REJECTED"
                fb = feedback_match.group(1).strip() if feedback_match else "No feedback provided"
                
                # Store the best approved fix
                if verdict == "APPROVED":
                    print(f"\n{model_name} fix approved!")
                    if best_fix is None:
                        best_fix = fixed_code
                        best_feedback = f"Approved fix from {model_name}"
                        # Don't break - continue to find all approved fixes
                
                # Update model ranking based on feedback
                feedback_summary = f"{model_name}: {verdict} - {fb[:100]}"
                model_ranking.append(feedback_summary)
            
            # Select best fix if any approved
            if best_fix:
                print("\n" + "="*80)
                print("FIX APPROVED BY VERIFIER!")
                print("="*80)
                return best_fix, "SUCCESS", defect_class
            
            # If no approved fix, select based on model consensus
            print("\nNo approved fix found. Selecting best candidate...")
            if model_ranking:
                # Prioritize models with successful execution
                successful_models = [m for m in model_ranking if isinstance(m, str) and "SUCCESS" in m]
                if successful_models:
                    selected_model = successful_models[0]
                    current_code = fixes[selected_model]
                    feedback = f"Selected {selected_model} fix (passed execution)"
                else:
                    # Fallback to first model's fix
                    first_model = list(fixes.keys())[0]
                    current_code = fixes[first_model]
                    feedback = f"Using {first_model} fix as fallback"
                
                # Add aggregated feedback
                feedback += "\nModel Feedback:\n- " + "\n- ".join(model_ranking)
            else:
                feedback = "All model fixes failed - using last generated code"
                current_code = list(fixes.values())[0] if fixes else current_code
            
            attempts += 1
            
        except Exception as e:
            print(f"System error during workflow: {str(e)}")
            feedback = f"System error: {str(e)}"
            attempts += 1
    
    # All attempts failed - return the latest generated code
    print("\n" + "="*80)
    print("ALL ATTEMPTS FAILED! SAVING LATEST GENERATED CODE")
    print("="*80)
    return current_code, "FAILED", f"Max attempts reached. Last feedback: {feedback}"

def process_programs():
    results = []
    
    if not os.listdir(INPUT_DIR):
        print(f"\nError: No Python files found in input directory: {INPUT_DIR}")
        return
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".py"):
            file_path = os.path.join(INPUT_DIR, filename)
            print(f"\n{'='*80}")
            print(f"PROCESSING FILE: {filename}")
            print("="*80)
            
            try:
                with open(file_path, "r") as f:
                    original_code = f.read()
                
                fixed_code, status, message = repair_workflow(original_code, filename)
                
                # Determine output filename based on status
                if status == "SUCCESS":
                    fixed_filename = f"{filename}"
                else:  # FAILED status
                    base_name = os.path.splitext(filename)[0]
                    fixed_filename = f"{base_name}_attempt_failed.py"
                
                fixed_path = os.path.join(OUTPUT_DIR, fixed_filename)
                with open(fixed_path, "w") as f:
                    f.write(fixed_code)
                print(f"\nSaved {'fixed' if status == 'SUCCESS' else 'latest'} version to: {fixed_path}")
                
                results.append((filename, status, message))
                
            except Exception as e:
                results.append((filename, "ERROR", str(e)))
                print(f"Processing failed: {str(e)}")
    
    # Print report
    print("\n\n" + "="*80)
    print("REPAIR SUMMARY")
    print("="*80)
    print("{:<30} {:<15} {:<50}".format("Filename", "Status", "Details"))
    print("-"*95)
    for filename, status, details in results:
        print("{:<30} {:<15} {:<50}".format(filename, status, details))
    print("="*95)

if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print(f"###{' QuixBugs Repair System ':-^74}###")
    print(f"###{' Input: ' + INPUT_DIR + ' ':-^74}###")
    print(f"###{' Output: ' + OUTPUT_DIR + ' ':-^74}###")
    print(f"###{' Test Cases: ' + TESTCASE_DIR + ' ':-^74}###")
    print(f"{'#'*80}\n")
    process_programs()
    print("\nOperation complete. Verify output directory for results.")
