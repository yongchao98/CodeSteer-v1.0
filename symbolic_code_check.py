def analyze_computational_approach(code_string):
    """
    Analyzes Python code to determine if it uses searching, symbolic computing,
    or numerical computing approaches.

    Returns a dictionary with analysis results and explanations.
    """
    import ast
    import re

    analysis = {
        'searching_patterns': False,
        'symbolic_computing': False,
        'numerical_computing': False,
        'explanations': []
    }

    # Parse the code into an AST
    try:
        tree = ast.parse(code_string)
    except Exception as e:
        return {
            'error': True,
            'message': f"Invalid Python code: {str(e)}",
            'searching_patterns': False,
            'symbolic_computing': False,
            'numerical_computing': False,
            'explanations': [],
            'complexity_score': 0
        }

    # Searching patterns indicators
    searching_indicators = {
        'itertools': ['permutations', 'combinations', 'product'],
        'loops': ['for', 'while'],
        'comprehensive_search': ['all', 'any', 'filter', 'map']
    }

    # Symbolic computing indicators
    symbolic_indicators = {
        'string_expressions': ['eval', 'exec'],
        'dynamic_expression_building': ['+', '-', '*', '/', '(', ')'],
        'string_formatting': ['format', 'f"', "f'"]
    }

    # Numerical computing indicators
    numerical_indicators = {
        'math_operations': ['math.', 'numpy.', 'scipy.'],
        'complex_calculations': ['sum', 'abs', 'pow', 'sqrt']
    }

    # Initialize complexity score
    complexity_score = 0

    # Check for searching patterns
    for node in ast.walk(tree):
        # Check for iterations and loops
        if isinstance(node, (ast.For, ast.While)):
            analysis['searching_patterns'] = True
            analysis['explanations'].append("Contains loops for systematic search")

        # Check for itertools usage
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module_name = node.names[0].name
            if 'itertools' in module_name:
                analysis['searching_patterns'] = True
                analysis['explanations'].append("Uses itertools for combinatorial search")

    # Check for symbolic computing
    if 'eval(' in code_string or 'exec(' in code_string:
        analysis['symbolic_computing'] = True
        analysis['explanations'].append("Uses eval/exec for symbolic expression evaluation")

    # Check for f-strings and string formatting
    for format_indicator in symbolic_indicators['string_formatting']:
        if format_indicator in code_string:
            analysis['symbolic_computing'] = True
            analysis['explanations'].append("Uses string formatting for expression building")
            break

    # Check for numerical computing and count mathematical operations
    math_ops_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            analysis['numerical_computing'] = True
            analysis['explanations'].append("Contains direct mathematical operations")
            math_ops_count += 1

    # Count complexity indicators
    complexity_score += len(re.findall(r'for|while', code_string))  # Full weight for loops
    complexity_score += len(re.findall(r'if|elif|else', code_string))  # Full weight for conditionals
    complexity_score += len(re.findall(r'try|except', code_string))  # Full weight for error handling
    complexity_score += math_ops_count * 0.25  # Half weight for mathematical operations

    analysis['complexity_score'] = complexity_score

    return analysis

def analyze_code_and_explain(code):
    """Analyzes code and provides a human-readable explanation."""
    results = analyze_computational_approach(code)

    if 'error' in results:
        return results['message'], results['complexity_score']

    explanation = []
    if results['searching_patterns']:
        explanation.append("SEARCHING APPROACH detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'search' in exp.lower()])

    if results['symbolic_computing']:
        explanation.append("\nSYMBOLIC COMPUTING detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'symbol' in exp.lower() or 'expression' in exp.lower()])

    if results['numerical_computing']:
        explanation.append("\nNUMERICAL COMPUTING detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'mathematical' in exp.lower()])

    explanation.append(f"\nComplexity score: {results['complexity_score']}")

    return "\n".join(explanation), results['complexity_score']