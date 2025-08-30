# Contributing to Diffusion-LLM-RS

Thank you for your interest in contributing to Diffusion-LLM-RS! We welcome contributions from the community to help improve this project.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Pull Request Process](#-pull-request-process)
- [Reporting Bugs](#-reporting-bugs)
- [Feature Requests](#-feature-requests)
- [License](#-license)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** the project to your own machine
3. **Create** a new branch for your changes
4. **Commit** changes to your branch
5. **Push** your work back up to your fork
6. Submit a **Pull Request** so we can review your changes

## 🔄 Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Your detailed description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request against the `main` branch

## 🎨 Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` to catch common mistakes
- Document public APIs with Rustdoc comments

## 🧪 Testing

- Write unit tests for new functionality
- Run all tests with `cargo test --all-features`
- Ensure all tests pass before submitting a PR
- Add integration tests for significant features

## 🔄 Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations, and container parameters
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/)
4. Your PR will be reviewed by the maintainers
5. Once approved, a maintainer will merge your PR

## 🐛 Reporting Bugs

Please open an issue with the following information:

- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Any relevant logs or error messages
- Your environment (OS, Rust version, etc.)

## 💡 Feature Requests

We welcome feature requests! Please open an issue with:

- A clear description of the feature
- The problem it solves
- Any alternative solutions you've considered
- Additional context or screenshots if applicable

## 📄 License

By contributing, you agree that your contributions will be licensed under its [Apache 2.0 License](LICENSE).
