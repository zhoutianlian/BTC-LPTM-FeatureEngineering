
(function () {
  function cellValue(row, idx) {
    return row.children[idx].innerText.trim();
  }
  function asNumber(value) {
    var cleaned = value.replace(/%$/, "");
    var n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
  }
  document.addEventListener("DOMContentLoaded", function () {
    var table = document.getElementById("featureTable");
    var search = document.getElementById("featureSearch");
    if (!table) return;
    Array.prototype.forEach.call(table.querySelectorAll("th"), function (th, idx) {
      th.addEventListener("click", function () {
        var tbody = table.tBodies[0];
        var rows = Array.prototype.slice.call(tbody.rows);
        var dir = th.dataset.dir === "asc" ? "desc" : "asc";
        th.dataset.dir = dir;
        rows.sort(function (a, b) {
          var av = cellValue(a, idx);
          var bv = cellValue(b, idx);
          var an = asNumber(av);
          var bn = asNumber(bv);
          var cmp = an !== null && bn !== null ? an - bn : av.localeCompare(bv);
          return dir === "asc" ? cmp : -cmp;
        });
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });
    if (search) {
      search.addEventListener("input", function () {
        var q = search.value.toLowerCase();
        Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
          row.style.display = row.innerText.toLowerCase().indexOf(q) >= 0 ? "" : "none";
        });
      });
    }
  });
})();
