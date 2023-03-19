exports.handler = function(context, event, callback) {
  
  let m = event.nitrogen;
  let x = event.phosphorus;
  let a = event.potassium;
  
  let ammoniumPhosphate = x *453.592 / 31.09 * 149.09
  let kcl = a * 453.592 /39.1*74.55;
  
  let nitrogenAlreadyAdded = ammoniumPhosphate/149.09*3;
  let nitrogenNeeded = m * 453.592/14;
  
  let nitrogen = nitrogenNeeded - nitrogenAlreadyAdded;
  
  let urea = nitrogen/2*60.06;
 
  let farmSize = event.farmSize
  
  /* Your code goes here */
  
  callback(null, {"ammoniumPhosphate": Math.round(ammoniumPhosphate * farmSize / 1000), "kcl": Math.round(kcl * farmSize/ 1000), "urea": Math.round(urea * farmSize / 1000)});
};