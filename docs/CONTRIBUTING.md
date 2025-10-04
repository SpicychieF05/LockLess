# Contributing to LockLess

Thank you for your interest in contributing to LockLess! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account
- Basic understanding of biometric authentication concepts

### Development Setup

1. **Fork the repository**

   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/LockLess.git
   cd LockLess
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/SpicychieF05/LockLess.git
   ```

4. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

5. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check `help wanted` for more complex tasks
- Create a new issue if you have a feature request or bug report

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clean, readable code
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run linting
flake8 src/ tests/
black --check src/ tests/
mypy src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `style:` for formatting changes
- `perf:` for performance improvements

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def process_biometric_data(data: np.ndarray, config: Dict[str, Any]) -> BiometricResult:
    """Process biometric data with configuration.

    Args:
        data: Input biometric data array
        config: Configuration dictionary

    Returns:
        Processed biometric result

    Raises:
        BiometricError: If processing fails
    """
    pass
```

### Code Organization

- **Single Responsibility**: Each function/class should have one clear purpose
- **DRY Principle**: Don't repeat yourself
- **Clear Naming**: Use descriptive variable and function names
- **Documentation**: Document all public APIs
- **Error Handling**: Use appropriate exception handling

### Security Guidelines

- **Never log sensitive data** (passwords, biometric templates)
- **Use secure random number generation**
- **Validate all inputs**
- **Follow OWASP guidelines**
- **Encrypt sensitive data at rest**

### Example Code Structure

```python
"""Module for biometric authentication."""

import logging
from typing import Optional, Dict, Any
import numpy as np

from src.core.exceptions import BiometricError
from src.core.config import ConfigManager

logger = logging.getLogger(__name__)


class BiometricProcessor:
    """Processes biometric data for authentication."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize biometric processor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._setup_models()

    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """Process biometric data.

        Args:
            data: Input biometric data

        Returns:
            Processing results

        Raises:
            BiometricError: If processing fails
        """
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"Biometric processing failed: {e}")
            raise BiometricError(f"Processing failed: {e}") from e
```

## Testing Requirements

### Test Categories

1. **Unit Tests** (`tests/unit/`)

   - Test individual functions and methods
   - Mock external dependencies
   - Aim for 90%+ code coverage

2. **Integration Tests** (`tests/integration/`)

   - Test component interactions
   - Use real dependencies where possible
   - Test API endpoints

3. **Biometric Tests** (`tests/biometric/`)

   - Test biometric accuracy (FAR/FRR)
   - Performance benchmarks
   - Dataset validation

4. **Security Tests** (`tests/security/`)
   - Penetration testing
   - Encryption validation
   - Anti-spoofing tests

### Writing Tests

```python
"""Tests for biometric authentication."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.biometric.authentication import AuthenticationEngine
from src.core.exceptions import BiometricError


class TestAuthenticationEngine:
    """Test cases for AuthenticationEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'similarity_threshold': 0.7,
            'quality_threshold': 0.6
        }
        self.engine = AuthenticationEngine(self.config)

    def test_authenticate_success(self):
        """Test successful authentication."""
        # Arrange
        user_id = "test_user"
        password = "test_password"
        mock_template = np.random.rand(512)

        with patch.object(self.engine, '_load_template', return_value=mock_template):
            with patch.object(self.engine, '_extract_features', return_value=mock_template):
                # Act
                result = self.engine.authenticate_user(user_id, password)

                # Assert
                assert result.success is True
                assert result.user_id == user_id

    def test_authenticate_invalid_user(self):
        """Test authentication with invalid user."""
        # Arrange
        user_id = "invalid_user"
        password = "test_password"

        with patch.object(self.engine, '_load_template', side_effect=FileNotFoundError):
            # Act & Assert
            with pytest.raises(BiometricError):
                self.engine.authenticate_user(user_id, password)

    @pytest.mark.performance
    def test_authentication_performance(self):
        """Test authentication performance."""
        # Performance test implementation
        pass
```

### Test Data

- Use synthetic data for unit tests
- Use anonymized real data for integration tests
- Never commit real biometric data to the repository

## Documentation

### Code Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for all functions
- **Comments**: Explain complex logic
- **README Updates**: Update README for new features

### API Documentation

- Document all public APIs
- Include examples
- Specify error conditions
- Update API reference

### Example Documentation

```python
def enroll_user(user_id: str, password: str, config: Optional[Dict] = None) -> EnrollmentResult:
    """Enroll a new user with biometric data.

    This function captures biometric samples from the user, processes them,
    and stores an encrypted template for future authentication.

    Args:
        user_id: Unique identifier for the user
        password: Master password for template encryption
        config: Optional configuration dictionary

    Returns:
        EnrollmentResult containing success status and metadata

    Raises:
        EnrollmentError: If enrollment fails
        ValidationError: If input validation fails

    Example:
        >>> result = enroll_user("john_doe", "secret123")
        >>> if result.success:
        ...     print(f"Enrolled user: {result.user_id}")
        ...     print(f"Samples: {result.samples_collected}")
    """
```

## Issue Guidelines

### Bug Reports

When reporting bugs, include:

1. **Clear Description**: What happened vs. what you expected
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Environment**: OS, Python version, dependencies
4. **Logs**: Relevant error messages and logs
5. **Screenshots**: If applicable

### Feature Requests

When requesting features, include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions considered
4. **Additional Context**: Any other relevant information

### Issue Templates

Use the provided issue templates:

- Bug report template
- Feature request template
- Security vulnerability template

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages follow conventions

### PR Description

Include:

1. **Summary**: Brief description of changes
2. **Type**: Bug fix, feature, documentation, etc.
3. **Testing**: How you tested the changes
4. **Breaking Changes**: Any breaking changes
5. **Screenshots**: If UI changes

### Review Process

1. **Automated Checks**: CI/CD pipeline runs
2. **Code Review**: At least one maintainer reviews
3. **Testing**: Manual testing if needed
4. **Approval**: Maintainer approves the PR
5. **Merge**: PR is merged to main branch

### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)

Add screenshots here
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update Version**: Update version in `setup.py` and `__init__.py`
2. **Update Changelog**: Add changes to `CHANGELOG.md`
3. **Create Release**: Create GitHub release with tag
4. **Publish**: Publish to PyPI (if applicable)

## Community Guidelines

### Communication

- **Be Respectful**: Treat everyone with respect
- **Be Constructive**: Provide helpful feedback
- **Be Patient**: Remember that everyone is learning
- **Be Professional**: Keep discussions professional

### Getting Help

- **Documentation**: Check existing documentation first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our Discord community

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Mentioned in release notes
- **Documentation**: Credited in relevant sections

## Questions?

If you have questions about contributing:

- Open a [GitHub Discussion](https://github.com/SpicychieF05/LockLess/discussions)
- Join our [Discord](https://discord.gg/BByDM7fZ)
- Contact maintainers directly

Thank you for contributing to LockLess! 🎉
