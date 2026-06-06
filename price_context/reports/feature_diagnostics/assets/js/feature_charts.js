
(function () {
  var payload = window.FEATURE_DIAGNOSTIC_PAYLOAD;
  if (!payload || !window.Plotly) return;

  var COLORS = {
    cyan: "#18E0FF",
    green: "#00FF88",
    orange: "#FF7A1A",
    yellow: "#F6C945",
    red: "#FF5A7A",
    purple: "#A46CFF",
    muted: "#7A8CA5",
    paper: "#07111F",
    plot: "#08111F",
    text: "#E7F6FF"
  };
  var CONFIG = { displaylogo: false, responsive: true, scrollZoom: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  function finite(v) {
    return typeof v === "number" && Number.isFinite(v);
  }
  function cleanValues(values) {
    return values.map(function (v) { return finite(v) ? v : null; });
  }
  function finiteValues(values) {
    return values.filter(finite);
  }
  function layout(title, extra) {
    var base = {
      title: { text: title, font: { color: COLORS.text, size: 17 } },
      template: "plotly_dark",
      paper_bgcolor: COLORS.paper,
      plot_bgcolor: COLORS.plot,
      font: { color: "#D8E7F5", family: "Inter, Segoe UI, Arial, sans-serif" },
      margin: { l: 58, r: 32, t: 76, b: 52 },
      hovermode: "closest",
      legend: {
        bgcolor: "rgba(7,17,31,0.76)",
        bordercolor: "rgba(130,180,255,0.22)",
        borderwidth: 1,
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "right",
        x: 1
      },
      xaxis: axisStyle(),
      yaxis: axisStyle()
    };
    return Object.assign(base, extra || {});
  }
  function axisStyle() {
    return {
      gridcolor: "rgba(170, 210, 255, 0.10)",
      zerolinecolor: "rgba(170, 210, 255, 0.16)",
      showline: true,
      linecolor: "rgba(170, 210, 255, 0.18)",
      rangemode: "normal"
    };
  }
  function timeTraceProps() {
    if (payload.time_axis && payload.time_axis.mode === "linear") {
      return { x0: payload.time_axis.x0, dx: payload.time_axis.dx };
    }
    return { x: payload.time_axis ? payload.time_axis.values : [] };
  }
  function timeForIndex(i) {
    if (payload.time_axis && payload.time_axis.mode === "linear") {
      return new Date(Date.parse(payload.time_axis.x0) + payload.time_axis.dx * i).toISOString();
    }
    return payload.time_axis.values[i];
  }
  function timeForIndices(indices) {
    return indices.map(timeForIndex);
  }
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    var pos = (sorted.length - 1) * q;
    var lo = Math.floor(pos);
    var hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function sortedFinite(values) {
    return finiteValues(values).sort(function (a, b) { return a - b; });
  }
  function mean(values) {
    var vals = finiteValues(values);
    if (!vals.length) return null;
    return vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
  }
  function std(values) {
    var vals = finiteValues(values);
    if (vals.length < 2) return 0;
    var m = mean(vals);
    var s = vals.reduce(function (a, b) { return a + Math.pow(b - m, 2); }, 0);
    return Math.sqrt(s / (vals.length - 1));
  }
  function minValue(values) {
    var out = null;
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && (out === null || values[i] < out)) out = values[i];
    }
    return out;
  }
  function maxValue(values) {
    var out = null;
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && (out === null || values[i] > out)) out = values[i];
    }
    return out;
  }
  function bisectLeft(arr, x) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  function addSorted(arr, x) {
    arr.splice(bisectLeft(arr, x), 0, x);
  }
  function removeSorted(arr, x) {
    var idx = bisectLeft(arr, x);
    if (idx < arr.length) arr.splice(idx, 1);
  }
  function rollingStats(values, window) {
    var n = values.length;
    var sorted = [];
    var sum = 0;
    var sumSq = 0;
    var count = 0;
    var minPeriods = Math.min(window, 10);
    var out = { mean: new Array(n), std: new Array(n), q05: new Array(n), q95: new Array(n), miss: new Array(n) };
    var missWindow = [];
    var missSum = 0;
    for (var i = 0; i < n; i++) {
      var v = values[i];
      var miss = finite(v) ? 0 : 1;
      missWindow.push(miss);
      missSum += miss;
      if (finite(v)) {
        addSorted(sorted, v);
        sum += v;
        sumSq += v * v;
        count += 1;
      }
      if (i >= window) {
        var old = values[i - window];
        var oldMiss = missWindow.shift();
        missSum -= oldMiss;
        if (finite(old)) {
          removeSorted(sorted, old);
          sum -= old;
          sumSq -= old * old;
          count -= 1;
        }
      }
      if (count >= minPeriods) {
        var m = sum / count;
        var variance = Math.max(0, (sumSq - sum * sum / count) / Math.max(1, count - 1));
        out.mean[i] = m;
        out.std[i] = Math.sqrt(variance);
        out.q05[i] = quantile(sorted, 0.05);
        out.q95[i] = quantile(sorted, 0.95);
      } else {
        out.mean[i] = null; out.std[i] = null; out.q05[i] = null; out.q95[i] = null;
      }
      out.miss[i] = missSum / Math.min(window, i + 1);
    }
    return out;
  }
  function outlierIndices(values) {
    var vals = finiteValues(values);
    if (vals.length < 3) return [];
    var m = mean(vals);
    var s = std(vals);
    if (!s) return [];
    var idx = [];
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && Math.abs((values[i] - m) / s) > payload.zscore_threshold) idx.push(i);
    }
    idx.sort(function (a, b) { return Math.abs(values[b] - m) - Math.abs(values[a] - m); });
    return idx.slice(0, payload.max_plot_outlier_points || 500);
  }
  function missingIndices(values) {
    var idx = [];
    for (var i = 0; i < values.length; i++) if (!finite(values[i])) idx.push(i);
    var max = payload.max_missing_markers || 500;
    if (idx.length <= max) return idx;
    var sampled = [];
    for (var j = 0; j < max; j++) sampled.push(idx[Math.floor(j * (idx.length - 1) / Math.max(1, max - 1))]);
    return sampled;
  }
  function histogram(values, bins) {
    var vals = finiteValues(values);
    if (!vals.length) return { x: [], y: [], smooth: [] };
    var min = minValue(vals);
    var max = maxValue(vals);
    if (min === max) return { x: [min], y: [1], smooth: [1] };
    var width = (max - min) / bins;
    var counts = new Array(bins).fill(0);
    vals.forEach(function (v) {
      var idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
      counts[idx] += 1;
    });
    var density = counts.map(function (c) { return c / vals.length / width; });
    var centers = counts.map(function (_, i) { return min + width * (i + 0.5); });
    var kernel = [1, 2, 3, 4, 3, 2, 1];
    var ksum = kernel.reduce(function (a, b) { return a + b; }, 0);
    var smooth = density.map(function (_, i) {
      var total = 0;
      for (var k = 0; k < kernel.length; k++) {
        var j = i + k - 3;
        if (j >= 0 && j < density.length) total += density[j] * kernel[k];
      }
      return total / ksum;
    });
    return { x: centers, y: density, smooth: smooth };
  }
  function renderTimeSeries(values) {
    var traces = [Object.assign({
      y: values,
      type: "scatter",
      mode: "lines",
      name: payload.feature_name,
      line: { color: COLORS.cyan, width: 1.2 },
      connectgaps: false,
      hovertemplate: "time=%{x}<br>value=%{y:.6g}<extra></extra>"
    }, timeTraceProps())];
    var outIdx = outlierIndices(values);
    if (outIdx.length) {
      traces.push({
        x: timeForIndices(outIdx),
        y: outIdx.map(function (i) { return values[i]; }),
        type: "scatter",
        mode: "markers",
        name: "marked outliers",
        marker: { color: COLORS.red, size: 6, symbol: "x" }
      });
    }
    var missIdx = missingIndices(values);
    if (missIdx.length) {
      var vals = finiteValues(values);
      var baseline = vals.length ? minValue(vals) : 0;
      traces.push({
        x: timeForIndices(missIdx),
        y: missIdx.map(function () { return baseline; }),
        type: "scatter",
        mode: "markers",
        name: "missing samples",
        marker: { color: COLORS.yellow, size: 4, symbol: "line-ns-open" }
      });
    }
    Plotly.newPlot("chart-time-series", traces, layout(payload.feature_name + " time series", {
      xaxis: Object.assign(axisStyle(), {
        rangeslider: { visible: true, bgcolor: COLORS.plot, bordercolor: "#1B2A3D" },
        rangeselector: {
          bgcolor: "#101C2C",
          activecolor: COLORS.cyan,
          font: { color: COLORS.text },
          buttons: [
            { count: 1, label: "1D", step: "day", stepmode: "backward" },
            { count: 7, label: "1W", step: "day", stepmode: "backward" },
            { count: 1, label: "1M", step: "month", stepmode: "backward" },
            { count: 3, label: "3M", step: "month", stepmode: "backward" },
            { count: 6, label: "6M", step: "month", stepmode: "backward" },
            { count: 1, label: "1Y", step: "year", stepmode: "backward" },
            { label: "ALL", step: "all" }
          ]
        }
      })
    }), CONFIG);
  }
  function renderDistribution(values) {
    var hist = histogram(values, 90);
    var vals = sortedFinite(values);
    var traces = [
      { x: hist.x, y: hist.y, type: "bar", name: "histogram", marker: { color: "rgba(24,224,255,0.62)", line: { color: COLORS.cyan, width: 0.5 } } },
      { x: hist.x, y: hist.smooth, type: "scatter", mode: "lines", name: "smoothed density", line: { color: COLORS.purple, width: 2 } }
    ];
    var shapes = [];
    var annotations = [];
    [["mean", mean(vals), COLORS.green], ["median", quantile(vals, 0.5), COLORS.yellow], ["p01", quantile(vals, 0.01), COLORS.orange], ["p99", quantile(vals, 0.99), COLORS.orange]].forEach(function (item) {
      if (!finite(item[1])) return;
      shapes.push({ type: "line", x0: item[1], x1: item[1], y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: item[2], dash: "dash", width: 1.4 } });
      annotations.push({ x: item[1], y: 1.02, xref: "x", yref: "paper", text: item[0], showarrow: false, font: { color: item[2], size: 11 } });
    });
    Plotly.newPlot("chart-distribution", traces, layout(payload.feature_name + " distribution", { shapes: shapes, annotations: annotations }), CONFIG);
  }
  function renderBox(values) {
    var vals = sortedFinite(values);
    if (!vals.length) {
      Plotly.newPlot("chart-box", [], layout(payload.feature_name + " box plot"), CONFIG);
      return;
    }
    var trace = {
      type: "box",
      name: payload.feature_name,
      q1: [quantile(vals, 0.25)],
      median: [quantile(vals, 0.5)],
      q3: [quantile(vals, 0.75)],
      lowerfence: [vals[0]],
      upperfence: [vals[vals.length - 1]],
      boxpoints: false,
      fillcolor: "rgba(24,224,255,0.32)",
      line: { color: COLORS.cyan }
    };
    Plotly.newPlot("chart-box", [trace], layout(payload.feature_name + " box plot"), CONFIG);
  }
  function renderRolling(values, stats) {
    var traces = [
      Object.assign({ y: values, type: "scatter", mode: "lines", name: "raw", line: { color: "rgba(24,224,255,0.50)", width: 1 } }, timeTraceProps()),
      Object.assign({ y: stats.mean, type: "scatter", mode: "lines", name: "rolling mean", line: { color: COLORS.green, width: 1.6 } }, timeTraceProps()),
      Object.assign({ y: stats.q05, type: "scatter", mode: "lines", name: "rolling q05", line: { color: COLORS.muted, width: 1, dash: "dot" } }, timeTraceProps()),
      Object.assign({ y: stats.q95, type: "scatter", mode: "lines", name: "rolling q95", line: { color: COLORS.muted, width: 1, dash: "dot" } }, timeTraceProps()),
      Object.assign({ y: stats.std, type: "scatter", mode: "lines", name: "rolling std", xaxis: "x2", yaxis: "y2", line: { color: COLORS.orange, width: 1.4 } }, timeTraceProps())
    ];
    Plotly.newPlot("chart-rolling", traces, layout(payload.feature_name + " rolling diagnostics", {
      grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
      xaxis2: Object.assign(axisStyle(), { rangeslider: { visible: true, bgcolor: COLORS.plot, bordercolor: "#1B2A3D" } }),
      yaxis2: axisStyle(),
      height: 640
    }), CONFIG);
  }
  function renderMissing(values, stats) {
    var missing = values.map(function (v) { return finite(v) ? 0 : 1; });
    var traces = [
      Object.assign({ y: missing, type: "scatter", mode: "markers", name: "missing indicator", marker: { color: COLORS.yellow, size: 3 } }, timeTraceProps()),
      Object.assign({ y: stats.miss, type: "scatter", mode: "lines", name: "rolling missing ratio", line: { color: COLORS.orange, width: 1.5 } }, timeTraceProps())
    ];
    Plotly.newPlot("chart-missing", traces, layout(payload.feature_name + " missing value timeline", { yaxis: Object.assign(axisStyle(), { range: [-0.05, 1.05] }) }), CONFIG);
  }
  function renderRelationship() {
    var rel = payload.relationship || {};
    var note = document.getElementById("relationship-note");
    if (note) note.innerText = window.FEATURE_RELATIONSHIP_NOTE || "";
    if (!rel.available || !rel.panels || !rel.panels.length) {
      var el = document.getElementById("chart-relationship");
      if (el) el.innerHTML = '<p class="note">Relationship plots are unavailable for this feature.</p>';
      return;
    }
    var traces = [];
    var layoutExtra = { grid: { rows: 1, columns: rel.panels.length, pattern: "independent" }, height: 470 };
    rel.panels.forEach(function (panel, idx) {
      var axisSuffix = idx === 0 ? "" : String(idx + 1);
      traces.push({
        x: panel.x,
        y: panel.y,
        type: "scatter",
        mode: "markers",
        name: panel.name,
        xaxis: "x" + axisSuffix,
        yaxis: "y" + axisSuffix,
        marker: { color: panel.color, size: 4, opacity: 0.38 },
        hovertemplate: payload.feature_name + "=%{x:.6g}<br>" + panel.y_name + "=%{y:.6g}<extra></extra>"
      });
      layoutExtra["xaxis" + axisSuffix] = Object.assign(axisStyle(), { title: payload.feature_name });
      layoutExtra["yaxis" + axisSuffix] = Object.assign(axisStyle(), { title: panel.y_name });
    });
    Plotly.newPlot("chart-relationship", traces, layout(payload.feature_name + " relationship diagnostics", layoutExtra), CONFIG);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var values = cleanValues(payload.values || []);
    renderTimeSeries(values);
    renderDistribution(values);
    renderBox(values);
    var stats = rollingStats(values, payload.rolling_window || 144);
    renderRolling(values, stats);
    renderMissing(values, stats);
    renderRelationship();
  });
})();
