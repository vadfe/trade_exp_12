# Project Rules

- The user runs the application and controls the runtime process. The coding agent may install/build in its own environment, run automated tests and short-lived verification processes, but must not present its background server as the user's project runtime.
- After every completed implementation stage, provide short but complete instructions covering:
  1. what was implemented;
  2. exact commands the user must run;
  3. how to verify the result in the UI or terminal;
  4. the expected successful result;
  5. any known blocker or limitation.
- Every pipeline stage must have an explicit UI visualization of its output, with explanations of what is shown and how to validate it. Never create a data artifact, feature set, label set, detector output, embedding or model report that is only stored on disk without a user-visible inspection page.
- User/server commands must use the project-local `.venv`. In the shared VMware workspace, the coding agent uses the project-local `.venv-devin` so it cannot corrupt or depend on the user's machine-specific `.venv`.
- Before declaring any stage complete, the coding agent must run pytest, ruff and real HTTP smoke-tests for every changed page/API against available local data. A unit-test-only result is not sufficient for UI work.
- A virtual environment is machine- and Python-version-specific. Never instruct the user to reuse `.venv` after changing Python versions or moving between machines.
