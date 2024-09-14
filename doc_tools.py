import re
import sys


def extract_classes(file_content: str):
    """
    Extracts class names and docstrings from the Python file content.

    Args:
        file_content (str): The content of the Python file as a string.

    Returns:
        List[Dict]: A list of dictionaries containing class information (name, docstring).
    """
    class_pattern = re.compile(
        r'@dataclass\s*\nclass\s+(\w+)\(.*?\):\s*"""(.*?)"""', re.S)
    classes = []

    for match in class_pattern.finditer(file_content):
        class_name = match.group(1)
        docstring = match.group(2).strip()
        classes.append({
            'name': class_name,
            'docstring': docstring
        })

    return classes


def extract_attribute_docs(docstring: str):
    """
    Extracts field descriptions from the 'Attributes:' section of the docstring.

    Args:
        docstring (str): The docstring of the class.

    Returns:
        Dict: A dictionary mapping field names to descriptions.
    """
    attributes_section = re.search(r'Attributes:\s*(.*)', docstring, re.S)

    if not attributes_section:
        return {}

    attributes_text = attributes_section.group(1)

    # Capture multi-line descriptions
    attribute_pattern = re.compile(
        r'(\w+)\s*\((.*?)\):\s*(.*?)(?=\n\s*\w+\s*\(|\n\s*$)', re.S)
    attribute_docs = {}

    for match in attribute_pattern.finditer(attributes_text):
        field_name = match.group(1)
        field_type = match.group(2)
        description = match.group(3).strip().replace('\n', ' ')
        attribute_docs[field_name] = {
            'type': field_type,
            'description': description
        }

    return attribute_docs


def generate_markdown(classes):
    """
    Generates markdown content from class information, including class docstrings and Attributes section.

    Args:
        classes (List[Dict]): A list of dictionaries containing class information.

    Returns:
        str: The generated markdown content.
    """
    md_content = ""

    for class_info in classes:
        class_name = class_info['name']
        docstring = class_info['docstring']

        # Include the class-level docstring as an introduction, removing the Attributes part
        clean_docstring = re.sub(r'Attributes:.*', '',
                                 docstring, flags=re.S).strip()

        md_content += f"## {class_name}\n\n"
        md_content += f"{clean_docstring}\n\n"

        # Extract attributes section from the docstring
        attribute_docs = extract_attribute_docs(docstring)

        for field_name, field_info in attribute_docs.items():
            field_type = field_info['type']
            description = field_info['description']

            md_content += f"- **{field_name}** (*{field_type}*): {
                description}\n"

        md_content += "\n"

    return md_content


if __name__ == '__main__':
    # Usage: python script.py input_file.py > output.md
    if len(sys.argv) != 2:
        print("Usage: python script.py input_file.py")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Process the file content to extract classes and generate markdown
    classes = extract_classes(file_content)
    markdown = generate_markdown(classes)

    # Print the markdown content (redirect this to a file if needed)
    print(markdown)
