// Ask the user to choose a directory to save images
saveDir = getDirectory("Choose a Directory to Save Images");

// Get the active image (stack)
id = getImageID();
selectImage(id);

// Convert the image to 8-bit grayscale
run("8-bit");

// Get the total number of slices in the stack
slices = nSlices;

// Loop through each slice and save
for (i = 1; i <= slices; i++) {
    // Select the i-th slice
    selectImage(id);
    setSlice(i);
    
    // Generate the file name (e.g., 000.jpg, 001.jpg, ...)
    fileName = IJ.pad(i-1, 3) + ".jpg";
    
    // Create the full path
    fullPath = saveDir + fileName;
    
    // Duplicate the current slice
    run("Duplicate...", "title=temp");
    
    // Save the duplicated slice as JPG
    saveAs("Jpeg", fullPath);
    
    // Close the duplicated slice
    close();
}

// Show a message when done
showMessage("Done", "All frames have been saved as jpg.");