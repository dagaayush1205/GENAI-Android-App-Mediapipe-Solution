import React from "react";
import "./App.css"; // Import the CSS file

function App() {
  return (
    <div className="app-container">
      <h1>Face & Hand Tracking</h1>
      <img src="http://localhost:5000/video_feed" alt="Video Stream" className="video-feed" />
    </div>
  );
}

export default App;
