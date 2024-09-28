const imageInput = document.getElementById("image-input");
const imageDisplay = document.getElementById("image-display");
const scanBeanButton = document.getElementById("scan-bean-button");
const importedImageContainer = document.getElementById(
  "imported-image-container"
);

imageInput.addEventListener("change", function () {
  const file = imageInput.files[0];
  const reader = new FileReader();

  reader.onload = function (event) {
    const img = new Image();
    img.src = event.target.result;
    imageDisplay.src = img.src;
    importedImageContainer.style.display = "block";

    if (file) {
      scanBeanButton.style.display = "block";
    } else {
      scanBeanButton.style.display = "none";
    }
  };

  reader.readAsDataURL(file);
});

scanBeanButton.addEventListener("click", function () {
  // Handle scan bean button click event
});
