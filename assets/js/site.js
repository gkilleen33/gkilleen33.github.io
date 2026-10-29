// Assemble email links at runtime so the address isn't sitting in the HTML for scrapers.
document.querySelectorAll("[data-email-user]").forEach(function (el) {
  var addr = el.dataset.emailUser + "@" + el.dataset.emailDomain;
  el.href = "mailto:" + addr;
  if (el.dataset.emailShow !== undefined) el.textContent = addr;
});

// Show / hide paper abstracts on the research page.
document.querySelectorAll(".abstract-toggle").forEach(function (btn) {
  var panel = document.getElementById(btn.getAttribute("aria-controls"));
  if (!panel) return;
  btn.addEventListener("click", function () {
    var open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", String(!open));
    panel.hidden = open;
  });
});

// Open an abstract automatically when linked to directly (e.g. research.html#risk-aversion).
if (location.hash) {
  var target = document.getElementById(location.hash.slice(1));
  var toggle = target && target.querySelector(".abstract-toggle");
  if (toggle && toggle.getAttribute("aria-expanded") === "false") toggle.click();
}
