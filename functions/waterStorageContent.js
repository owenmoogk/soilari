exports.handler = function(context, event, callback) {

  let density = event.density / 100
  let clay = event.clay / 1000;
  let sand = event.sand / 1000;
  let silt = event.silt / 1000;

  let g = 14.01 + 0.03*(silt * clay) - 8.47 * density
  let wsc = 14.04 + 1.07*g - 9.46*density + 0.14*sand

  callback(null, {"WSC": Math.round(wsc * 1000) / 1000});
};