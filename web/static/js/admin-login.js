// Admin login interactions.

(function () {
    'use strict';

    var form = document.getElementById('login-form');
    var usernameInput = document.getElementById('username');
    var passwordInput = document.getElementById('password');
    var passwordToggle = document.getElementById('password-toggle');
    var loginButton = document.getElementById('login-btn');
    var buttonContent = document.getElementById('btn-text');
    var errorMessage = document.getElementById('error-message');
    var successMessage = document.getElementById('success-message');

    function initializeTheme() {
        var savedTheme = null;
        try {
            savedTheme = window.localStorage.getItem('adminDarkMode');
        } catch {
            savedTheme = null;
        }

        var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (savedTheme === 'true' || (savedTheme === null && prefersDark)) {
            document.documentElement.setAttribute('data-theme', 'dark');
        }
    }

    function setMessage(element, message) {
        element.textContent = message;
        element.hidden = !message;
    }

    function setSubmitting(isSubmitting) {
        loginButton.disabled = isSubmitting;
        form.setAttribute('aria-busy', String(isSubmitting));
        buttonContent.replaceChildren();
        if (isSubmitting) {
            var spinner = document.createElement('span');
            spinner.className = 'spinner';
            spinner.setAttribute('aria-hidden', 'true');
            buttonContent.appendChild(spinner);
            loginButton.setAttribute('aria-label', 'Signing in');
        } else {
            buttonContent.textContent = 'Sign In';
            loginButton.removeAttribute('aria-label');
        }
    }

    passwordToggle.addEventListener('click', function () {
        var isVisible = passwordInput.type === 'text';
        passwordInput.type = isVisible ? 'password' : 'text';
        passwordToggle.setAttribute('aria-pressed', String(!isVisible));
        passwordToggle.setAttribute('aria-label', isVisible ? 'Show password' : 'Hide password');
    });

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        setMessage(errorMessage, '');
        setMessage(successMessage, '');
        setSubmitting(true);

        try {
            var response = await window.fetch('/admin/login', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: usernameInput.value.trim(),
                    password: passwordInput.value
                })
            });
            var data = await response.json().catch(function () {
                return {};
            });
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Login failed. Please try again.');
            }

            setMessage(successMessage, 'Login successful. Redirecting...');
            window.setTimeout(function () {
                window.location.assign('/admin');
            }, 500);
        } catch (error) {
            setMessage(errorMessage, error.message || 'Login failed. Please try again.');
            setSubmitting(false);
        }
    });

    initializeTheme();
})();
