# Project RAG System
This project is begin developed suitable to SOLID principles and clean code practices. The codebase is organized into modules and classes that encapsulate specific functionality, making it easier to maintain and extend. The project also includes comprehensive documentation.

## Coding Standards
- Do not over-abstract. SOLID principles should guide structure, not force unnecessary layers. Avoid creating wrapper classes, extra modules, or indirection solely for the sake of abstraction. Only introduce new abstractions when they provide clear, tangible value.
- Use descriptive variable and function names that clearly indicate their purpose.
- Follow consistent indentation and spacing for readability.
- Include docstrings for all classes and functions to explain their behavior and usage.
- Avoid code duplication by creating reusable functions and classes.
- Handle exceptions gracefully and provide meaningful error messages.
- Handle major edge cases and validate inputs to prevent unexpected behavior. Create private helper methods for validation and edge case handling to keep the main logic clean and focused. If that validation is being or will be used more than once in the codebase, consider creating a separate utility module for it.
