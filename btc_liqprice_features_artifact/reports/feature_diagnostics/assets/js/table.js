
(function () {
  function parseValue(text) {
    var cleaned = text.trim().replace(/,/g, '');
    if (cleaned === 'true') return 1;
    if (cleaned === 'false') return 0;
    var num = Number(cleaned);
    return Number.isFinite(num) ? num : text.toLowerCase();
  }
  function sortTable(table, column, asc) {
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var av = parseValue(a.cells[column].innerText);
      var bv = parseValue(b.cells[column].innerText);
      if (av < bv) return asc ? -1 : 1;
      if (av > bv) return asc ? 1 : -1;
      return 0;
    });
    rows.forEach(function (row) { tbody.appendChild(row); });
  }
  document.querySelectorAll('table[data-sortable="true"]').forEach(function (table) {
    table.querySelectorAll('th').forEach(function (th, idx) {
      th.addEventListener('click', function () {
        var asc = th.getAttribute('data-asc') !== 'true';
        table.querySelectorAll('th').forEach(function (h) { h.removeAttribute('data-asc'); });
        th.setAttribute('data-asc', asc ? 'true' : 'false');
        sortTable(table, idx, asc);
      });
    });
  });
  document.querySelectorAll('[data-table-search]').forEach(function (input) {
    var selector = input.getAttribute('data-table-search');
    var table = document.querySelector(selector);
    if (!table) return;
    input.addEventListener('input', function () {
      var needle = input.value.toLowerCase();
      Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
        row.style.display = row.innerText.toLowerCase().indexOf(needle) >= 0 ? '' : 'none';
      });
    });
  });
})();
