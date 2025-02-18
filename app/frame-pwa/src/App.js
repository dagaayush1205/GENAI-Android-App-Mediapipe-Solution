
import React, { useState } from "react";
import "./App.css";

function App() {
  const [selectedOptions, setSelectedOptions] = useState({
    faceTracking: true,
    handTracking: true,
  });

  // Function to send data to Flask
  const sendCheckboxData = (updatedOptions) => {
    fetch("http://localhost:5000/update_options", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedOptions),
    })
      .then((response) => response.json())
      .then((data) => console.log("Response from Flask:", data))
      .catch((error) => console.error("Error:", error));
  };

  // Handle checkbox change
  const handleCheckboxChange = (event) => {
    const updatedOptions = {
      ...selectedOptions,
      [event.target.name]: event.target.checked,
    };

    setSelectedOptions(updatedOptions); // Update UI
    sendCheckboxData(updatedOptions); // Send to Flask
  };

  return (
    <div className="app-container">
      <h1>Face & Hand Tracking</h1>

      {/* Checkboxes */}
      <div className="checkbox-container">
        <label>
          <input
            type="checkbox"
            name="faceTracking"
            checked={selectedOptions.faceTracking}
            onChange={handleCheckboxChange}
          />
          Enable Face Tracking
        </label>

        <label>
          <input
            type="checkbox"
            name="handTracking"
            checked={selectedOptions.handTracking}
            onChange={handleCheckboxChange}
          />
          Enable Hand Tracking
        </label>
      </div>

      {/* Video Feed */}
      <img
        src="http://localhost:5000/video_feed"
        alt="Video Stream"
        className="video-feed"
      />
    </div>
  );
}



export default App;
