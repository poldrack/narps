import importlib
import os
import inspect
from radon.raw import analyze
import ast


def print_tree(d, prefix=''):
    """Recursively print a tree structure from a nested dictionary with connecting lines and values."""
    # Get the list of keys to determine the last element
    keys = list(d.keys())
    for i, key in enumerate(keys):
        # Determine the connector based on whether it's the last element
        connector = '└── ' if i == len(keys) - 1 else '├── '
        if '__init__' in key:
            continue
        # Get the value associated with the key
        value = d[key]
        # Print the current key with its value
        if isinstance(value, dict):
            print(prefix + connector + f"{key}")
            # Adjust the prefix for the next level
            new_prefix = prefix + ('    ' if i == len(keys) - 1 else '│   ')
            print_tree(value, new_prefix)
        else:
            print(prefix + connector + f"{key} ({value})")


def get_python_files(sourcedir):
    for root, dirs, files in os.walk(sourcedir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                yield filepath


def get_class_definitions(sourcedir):
    # look at all files in the sourcedir and find all class definitions
    class_definitions = {}
    for file in get_python_files(sourcedir):
        with open(file, 'r') as f:
            content = f.read()
            try:
                tree = ast.parse(content)
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                if classes:
                    class_definitions[file] = classes
            except: # noqa
                # fails on files without a class definition
                pass

    return class_definitions


def get_class_info(class_definitions):
    class_info = {}
    for filepath, class_names in class_definitions.items():
        # Derive the module name from the file path
        # This assumes the file path is in a format like '/path/to/module.py'
        module_name = filepath.split('src/')[1].replace('/', '.').replace('.py', '')
        module = importlib.import_module(module_name)
        class_info[module_name] = {}
        for class_name in class_names:
            print('-'*20)
            cls = getattr(module, class_name)
            print('Class:', cls.__name__)
            class_info[module_name][cls.__name__] = {}
            class_members = inspect.getmembers(cls)
            # remove members with dunder names
            class_members = [m for m in class_members if '__' not in m[0]]

            for member in class_members:
                loc = analyze(inspect.getsource(member[1])).loc
                print(f'{member[0]}: {loc}')
                class_info[module_name][cls.__name__][member[0]] = loc
        print('-'*20)

    return class_info


def get_function_definitions(sourcedir):
    # look at all files in the sourcedir and find all class definitions

    function_definitions = {}
    for file in get_python_files(sourcedir):
        with open(file, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) 
                            if isinstance(node, ast.FunctionDef)]

            function_definitions[file] = functions

    return function_definitions


def get_function_info(function_definitions, verbose=False):
    function_info = {}
    for filepath, function_names in function_definitions.items():
        # Derive the module name from the file path
        # This assumes the file path is in a format like '/path/to/module.py'
        module_name = filepath.split('src/')[1].replace('/', '.').replace('.py', '')
        if verbose:
            print(module_name)
        try: 
            module = importlib.import_module(module_name)
        except Exception as e:
            if verbose:
                print(f'Error importing module {module_name}: {e}')
                print('-'*20)
            continue
        function_info[module_name] = {}
        for function_name in function_names:
            try:
                function = getattr(module, function_name)
                loc = analyze(inspect.getsource(function)).loc
                if verbose:
                    print(f'{function_name}: {loc}')
                function_info[module_name][function_name] = loc
            except Exception as e:
                if verbose:
                    print(f'Error processing {function_name}: {e}')
        if verbose:
            print('-'*20)
    return function_info
