// shs-main.js
window.showToast = function (message, type = "success") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast align-items-center text-bg-${type} border-0 shadow-lg`;
    toast.role = "alert";
    toast.ariaLive = "assertive";
    toast.ariaAtomic = "true";

    toast.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="toast-icon bi fs-4 me-3"></i>
            <div class="toast-body flex-grow-1">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>`;

    container.appendChild(toast);
    new bootstrap.Toast(toast, { delay: 3500 }).show();
};
