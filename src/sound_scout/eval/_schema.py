"""Schema errors of the eval files without their values (R09, R11).

jsonschema's `error.message` quotes the checked value: an annotator's text, a corpus name, a whole
manifest object. The message here is built from the JSON path and the violated rule alone.
"""

from jsonschema import Draft202012Validator


def schema_errors(schema: dict, instance: object) -> list[str]:
    """Every schema violation as `<json path>: <rule>`, sorted by path."""
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=_json_path)
    return [f"{error.json_path}: {schema_message(error)}" for error in errors]


def _json_path(error) -> str:
    return error.json_path


def schema_message(error) -> str:
    rule = error.validator_value
    match error.validator:
        case "required":
            return "; ".join(
                f"{name!r} is a required property" for name in rule if name not in error.instance
            )
        case "additionalProperties":
            allowed = sorted(error.schema["properties"])
            return f"Additional properties are not allowed, allowed: {allowed}"
        case "const":
            return f"{rule!r} was expected"
        case "enum":
            return f"must be one of {rule}"
        case "type":
            return f"must be of type {rule}"
        case "pattern":
            return f"must match pattern {rule!r}"
        case "minimum":
            return f"must be at least {rule}"
        case "exclusiveMinimum":
            return f"must be greater than {rule}"
        case "minLength":
            return f"must be at least {rule} characters long"
        case "uniqueItems":
            return "has non-unique items, every item must be unique"
        case "oneOf":
            return "must be " + " or ".join(_option(option) for option in rule)
    return f"violates schema rule {error.validator!r}"


def _option(schema: dict) -> str:
    if "$ref" in schema:
        return "a valid " + schema["$ref"].rsplit("/", 1)[-1].replace("_", " ")
    if isinstance(schema.get("type"), str):
        return schema["type"]
    return "a value of another allowed shape"
