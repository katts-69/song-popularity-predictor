// Where the Flask API lives. Locally it runs on port 5000; online it runs on Render.
window.SONG_API_BASE = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://localhost:5000"
  : "https://song-popularity-api-key9.onrender.com";
