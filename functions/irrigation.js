exports.handler = function(context, event, callback) {
  let x = {"maxWater": Math.round(event.farmSize * 1000 * event.wsc * event.depth * 100) / 100}
callback(null, x );
};