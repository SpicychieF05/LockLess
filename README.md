# Lockless - Privacy-First Biometric Authentication System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Android-blue)](https://github.com/lockless-auth)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/lockless-auth)

## Overview

Lockless is a locally-run biometric authentication system designed to rival Windows Hello while maintaining complete privacy and security. All processing happens offline with no cloud dependencies.

## Key Features

- 🔐 **Privacy-First**: 100% offline processing, encrypted template storage
- ⚡ **Fast**: Sub-500ms authentication latency
- 🛡️ **Secure**: AES-256 encryption, TPM/Secure Enclave integration
- 👤 **Multi-User**: Support for multiple enrolled users
- 🎯 **Accurate**: FAR ≤ 0.001%, FRR ≤ 1%
- 🔄 **Fallback**: PIN/password backup authentication
- 🌐 **Cross-Platform**: Windows, Linux, Android support
- 🔌 **API/SDK**: Third-party integration capabilities

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lockless-auth/lockless.git
cd lockless

# Install dependencies
pip install -r requirements.txt

# Run enrollment
python src/main.py --enroll --user "john_doe"

# Run authentication
python src/main.py --authenticate
```

## Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed information.

## Development Roadmap

See [docs/DEVELOPMENT_ROADMAP.md](docs/DEVELOPMENT_ROADMAP.md) for phased development plan.

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Security Architecture](docs/SECURITY.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Testing Strategy](docs/TESTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.