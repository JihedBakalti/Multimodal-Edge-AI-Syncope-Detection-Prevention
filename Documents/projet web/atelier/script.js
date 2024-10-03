var btn2 = document.getElementById("btn1");
btn2.addEventListener("click", function() {
    window.alert("button 1 clicked!");
  });

var btn = document.getElementById("btn2");

btn.addEventListener("click", function() {
  window.alert("button 2 clicked!");
});

var x = document.getElementById("Select");
x.addEventListener("change", function() {
    var value = document.getElementById("Select").value;
    document.getElementById("classe").innerHTML = "tu as selectionne: "+value;
})

document.getElementById("form").addEventListener("submit", function(e){
    e.preventDefault();
    var nom = document.getElementById("nom").value;
    document.getElementById("para").innerHTML="bienvenue :"+nom;
})