import re


def parse_dataclass(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    class_prop = {}
    current_comment = None

    class_pattern = re.compile(r'^class\s+(\w+)\(')
    field_pattern = re.compile(r'^\s*(\w+):\s*([\w\[\],\'"]+)\s*=\s*(.+)\s*$')
    comment_pattern = re.compile(r'^\s*#\s*(.+)$')

    for line in lines:
        class_match = class_pattern.match(line)
        if class_match:
            class_name = class_match.group(1)
            class_prop[class_name] = []
            continue

        comment_match = comment_pattern.match(line)
        if comment_match:
            current_comment = comment_match.group(1)
            continue

        field_match = field_pattern.match(line)
        if field_match:
            field_name = field_match.group(1)
            field_type = field_match.group(2)
            default_value = field_match.group(3)

            class_prop[class_name].append({
                'name': field_name,
                'type': field_type,
                'default': default_value,
                'comment': current_comment
            })
            current_comment = None
    class_prop.pop('ModelArgs')
    return class_prop


def to_markdown(data):
    markdown = ""
    for class_name, fields in data.items():
        markdown += f"# {class_name}\n\n"
        for field in fields:
            name = field['name']
            field_type = field['type']
            default = field['default']
            comment = field['comment']

            # Format each field
            markdown += f"""- **{name}** (*{field_type}*): {
                comment}. Default: `{default}`.\n"""
        markdown += "\n"  # Add a newline after each class
    return markdown


if __name__ == '__main__':
    # Example usage:
    file_path = 'tabular_transformer/hyperparameters.py'
    class_prop = parse_dataclass(file_path)
    md = to_markdown(class_prop)
    with open("docs/hyperparameters.md", "w") as file:
        file.write(md)
