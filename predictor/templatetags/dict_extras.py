# predictor/templatetags/dict_extras.py

from django import template

register = template.Library()

@register.filter
def dict_get(d, key):
    """
    Returns the value of a dictionary for the given key.
    Usage in template:
    {{ my_dict|dict_get:"key_name" }}
    """
    if isinstance(d, dict):
        return d.get(key, "")
    return ""