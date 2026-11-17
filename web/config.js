// Where the Flask API lives. Locally it runs on port 5000; online it runs on Render.
// TODO (step 10): replace the Render address once the API is deployed.
window.SONG_API_BASE = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://localhost:5000"
  : "https://song-popularity-api.onrender.com";
