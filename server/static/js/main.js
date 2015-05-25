// ugh, javascript
if (!String.prototype.format) {
  String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) {
      return typeof args[number] != 'undefined'
        ? args[number]
        : match
      ;
    });
  };
}

// Credit: <https://stackoverflow.com/a/7128796>
var percent_colors = [
    { pct: 0.0, color: { r: 0x31, g: 0x70, b: 0x8F } },
    { pct: 1.0, color: { r: 0x0A, g: 0xCA, b: 0x5C } } ];

var color_for_percent = function(pct) {
    for (var i = 1; i < percent_colors.length - 1; i++) {
        if (pct < percent_colors[i].pct) {
            break;
        }
    }
    var lower = percent_colors[i - 1];
    var upper = percent_colors[i];
    var range = upper.pct - lower.pct;
    var rangePct = (pct - lower.pct) / range;
    var pctLower = 1 - rangePct;
    var pctUpper = rangePct;
    var color = {
        r: Math.floor(lower.color.r * pctLower + upper.color.r * pctUpper),
        g: Math.floor(lower.color.g * pctLower + upper.color.g * pctUpper),
        b: Math.floor(lower.color.b * pctLower + upper.color.b * pctUpper)
    };
    return 'rgb(' + [color.r, color.g, color.b].join(',') + ')';
}
