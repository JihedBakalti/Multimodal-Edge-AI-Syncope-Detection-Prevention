var titleElement = document.getElementById("T1");
var destinationElement = document.getElementById("destination");

titleElement.addEventListener("keyup", function() {
    var titleValue = titleElement.value;
    var titleError = document.getElementById("title_error");

    if (titleValue.length < 3) {
        titleError.innerHTML = "le titre doit contenir au moins 3 caracterès";
        titleError.style.color = "red";
    } else {
        titleError.innerHTML = "Correct";
        titleError.style.color = "green";
    }
});

destinationElement.addEventListener("keyup", function() {
    var destinationValue = destinationElement.value;
    var destinationError = document.getElementById("destination_error");
    var pattern = /^[A-Za-z\s]{3,}$/; // Corrected pattern

    if (!pattern.test(destinationValue)) {
        destinationError.innerHTML = "La destination doit contenir uniquement des lettres et des espaces, et au moins 3 caractères";
        destinationError.style.color = "red";
    } else {
        destinationError.innerHTML = "Correct";
        destinationError.style.color = "green";
    }
});

