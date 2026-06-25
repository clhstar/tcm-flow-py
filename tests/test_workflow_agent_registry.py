import builtins
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from app.agents.workflow_agent.agent import WorkflowAgent


_MISSING = object()


@contextmanager
def preserved_modules(*module_names):
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in module_names}
    previous_parent_attrs = {}

    for name in module_names:
        parent_name, _, attr_name = name.rpartition(".")
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None:
            previous_parent_attrs[name] = (
                parent_module,
                attr_name,
                getattr(parent_module, attr_name, _MISSING),
            )

    try:
        yield
    finally:
        for name, module in previous_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        for parent_module, attr_name, attr_value in previous_parent_attrs.values():
            if attr_value is _MISSING:
                if hasattr(parent_module, attr_name):
                    delattr(parent_module, attr_name)
            else:
                setattr(parent_module, attr_name, attr_value)


class WorkflowAgentRegistryTests(unittest.TestCase):
    def test_preserved_modules_restores_parent_package_attribute(self):
        import app.agents as agents_package

        original_module = sys.modules.get("app.agents.registry", _MISSING)
        original_attr = getattr(agents_package, "registry", _MISSING)

        try:
            sys.modules.pop("app.agents.registry", None)
            if hasattr(agents_package, "registry"):
                delattr(agents_package, "registry")

            with preserved_modules("app.agents.registry"):
                importlib.import_module("app.agents.registry")
                self.assertTrue(hasattr(agents_package, "registry"))

            self.assertFalse(hasattr(agents_package, "registry"))
        finally:
            if original_module is _MISSING:
                sys.modules.pop("app.agents.registry", None)
            else:
                sys.modules["app.agents.registry"] = original_module

            if original_attr is _MISSING:
                if hasattr(agents_package, "registry"):
                    delattr(agents_package, "registry")
            else:
                setattr(agents_package, "registry", original_attr)

    def test_workflow_agent_resolution_does_not_import_lead_agent(self):
        real_import = builtins.__import__

        def fail_on_lead_agent_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("app.agents.lead_agent"):
                raise AssertionError(f"workflow_agent resolution imported {name}")
            return real_import(name, globals, locals, fromlist, level)

        with preserved_modules(
            "app.agents.registry",
            "app.agents.lead_agent",
            "app.agents.lead_agent.agent",
        ):
            sys.modules.pop("app.agents.registry", None)
            sys.modules.pop("app.agents.lead_agent", None)
            sys.modules.pop("app.agents.lead_agent.agent", None)
            import_patch = patch("builtins.__import__", side_effect=fail_on_lead_agent_import)

            with import_patch:
                registry = importlib.import_module("app.agents.registry")
                workflow_factory = registry.resolve_agent_factory("workflow_agent")

        self.assertEqual(workflow_factory.__name__, "make_workflow_agent")

    def test_resolves_workflow_agent_without_replacing_lead_agent(self):
        with preserved_modules(
            "app.agents.registry",
            "app.agents.lead_agent",
            "app.agents.lead_agent.agent",
        ):
            registry = importlib.import_module("app.agents.registry")
            workflow_factory = registry.resolve_agent_factory("workflow_agent")
            with patch("app.agents.workflow_agent.agent.build_workflow_model") as build_model:
                build_model.return_value.with_structured_output.return_value = object()
                workflow_agent = workflow_factory({})

        self.assertIsInstance(workflow_agent, WorkflowAgent)

        stub_lead = ModuleType("app.agents.lead_agent.agent")

        def make_lead_agent(config=None):
            return config

        stub_lead.make_lead_agent = make_lead_agent

        with preserved_modules("app.agents.lead_agent", "app.agents.lead_agent.agent"):
            with patch.dict(sys.modules, {"app.agents.lead_agent.agent": stub_lead}):
                lead_factory = registry.resolve_agent_factory("lead_agent")

        self.assertEqual(lead_factory.__name__, "make_lead_agent")

    def test_unknown_assistant_still_fails(self):
        registry = importlib.import_module("app.agents.registry")

        with self.assertRaisesRegex(ValueError, "^Unknown assistant_id: missing_agent$"):
            registry.resolve_agent_factory("missing_agent")


if __name__ == "__main__":
    unittest.main()
