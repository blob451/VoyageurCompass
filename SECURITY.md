# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in VoyageurCompass, please report it responsibly:

1. **Do not** create a public GitHub issue
2. Email security issues to the project maintainers
3. Include detailed information about the vulnerability
4. Allow reasonable time for investigation and fixes

## Security Scanning Results

This project uses automated security scanning tools:

### Bandit SAST Analysis
- **Current Status**: ✅ No critical or high severity issues
- **Low Severity Findings**: 5 (documented as acceptable)
- **Last Scan**: August 2025

#### Documented Acceptable Findings:
1. **B404 - subprocess import**: Used in test utilities (`scripts/test_*.py`)
   - **Risk Level**: Low
   - **Justification**: Test performance benchmarking scripts
   - **Mitigation**: Scripts are not part of production code

### Dependency Scanning
- Python dependencies scanned with `pip-audit`
- Node.js dependencies scanned with `npm audit`
- Regular updates applied for security patches

## Security Measures

### Authentication & Authorization
- JWT tokens with 60-minute access lifetime
- 7-day refresh token lifetime
- Token rotation and blacklisting enabled
- Secure password hashing with Django's built-in hashers

### Data Protection
- Database connection security
- Redis connection with authentication
- CORS properly configured
- Security middleware enabled

### Infrastructure Security
- Docker containerization with security best practices
- Multi-stage builds to minimize attack surface
- No secrets in container images
- Security headers enabled

## Security Guidelines for Contributors

1. **Never commit secrets or credentials**
2. **Use environment variables for configuration**
3. **Follow secure coding practices**
4. **Run security scans before submitting PRs**
5. **Keep dependencies updated**

## Contact

For security-related questions or concerns, please contact the project maintainers.