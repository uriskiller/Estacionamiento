function validar() {
var tarjeta = document.getElementById('id_tarjeta').value;
var ccv = document.getElementById('id_ccv').value;
if (tarjeta.length=16) {
  alert("OK");
}else{
  alert("No")
}
}
