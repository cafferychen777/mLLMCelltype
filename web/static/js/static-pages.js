// Shared behavior for static documentation pages.

(function () {
    'use strict';

    function toggleQuestion(heading) {
        var item = heading.closest('.faq-item');
        if (!item) return;
        var expanded = item.classList.toggle('active');
        heading.setAttribute('aria-expanded', String(expanded));
    }

    // FAQ questions expand without requiring a framework.
    document.querySelectorAll('.faq-item h3').forEach(function (heading) {
        heading.setAttribute('role', 'button');
        heading.setAttribute('tabindex', '0');
        heading.setAttribute('aria-expanded', 'false');
    });

    document.addEventListener('click', function (e) {
        var heading = e.target.closest('.faq-item h3');
        if (!heading) return;
        toggleQuestion(heading);
    });

    document.addEventListener('keydown', function (e) {
        var heading = e.target.closest('.faq-item h3');
        if (!heading || (e.key !== 'Enter' && e.key !== ' ')) return;
        e.preventDefault();
        toggleQuestion(heading);
    });
})();
