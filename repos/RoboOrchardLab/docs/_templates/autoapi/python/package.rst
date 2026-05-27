{% if obj.display and is_own_page %}
{# Renders the page title, e.g., "my_package" #}
{{ obj.short_name }}
{{ "=" * obj.short_name|length }}

{# Sets the python domain context for this page #}
.. py:module:: {{ obj.name }}

{# Section 1: Render the package's own docstring (from __init__.py) #}
{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

{% endif %}

{# Section 2: Render the list of submodules and subpackages #}
{% block submodules %}
   {# First, get all visible sub-packages and sub-modules #}
   {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
   {% set visible_submodules = obj.submodules|selectattr("display")|list %}

   {# Combine them into a single list and sort alphabetically #}
   {% set all_submodules = (visible_subpackages + visible_submodules)|sort(attribute='name') %}

   {# If the list is not empty, create a toctree for them #}
   {% if all_submodules %}
Submodules
----------

.. toctree::
   :maxdepth: 1

      {% for submodule in all_submodules %}
   {{ submodule.include_path }}
      {% endfor %}

   {% endif %}
{% endblock %}

{# All other content, like summary tables for classes, functions, etc., has been removed. #}
{% endif %}
