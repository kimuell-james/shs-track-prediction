// django-toasts.js
document.addEventListener("DOMContentLoaded", function () {
    // Check if Django messages were injected
    const toastMessages = window.djangoMessages || [];
    if (typeof showToast !== "function") {
        console.warn("showToast() not found — make sure shs-main.js is loaded first!");
        return;
    }

    // Loop through Django messages and display as toasts
    toastMessages.forEach(msg => {
        showToast(msg.text, msg.type);
    });
});
