import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [apiKey, setApiKey] = useState(""); // State to store the API key
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [tokens, setTokens] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleApiKeyChange = (event) => {
    setApiKey(event.target.value);
  };

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("question", question);
    formData.append("apikey", apiKey);

    try {
      const response = await fetch("http://localhost:8000/ask_file", {
        method: "POST",
        body: formData,
        headers: {
          Authorization: `Bearer ${apiKey}`, // Pass the API key in the request header
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAnswer(data.answer);
      setTokens(data.tokens);
    } catch (error) {
      console.error("Error:", error);
      setAnswer("An error occurred while processing your request.");
    }
  };

  return (
    <div>
      <h1>Speech Embedding App</h1>
      <form onSubmit={handleSubmit} encType="multipart/form-data">
        <div>
          <label htmlFor="file">Upload PDF or Text File:</label>
          <input
            type="file"
            id="file"
            name="file"
            accept=".pdf, .txt"
            onChange={handleFileChange}
          />
        </div>
        <div>
          <label htmlFor="apiKey">Enter your API Key:</label>
          <input
            type="text"
            id="apiKey"
            value={apiKey}
            onChange={handleApiKeyChange}
          />
        </div>
        <div>
          <label htmlFor="question">Enter your question:</label>
          <input
            type="text"
            name="question"
            id="question"
            value={question}
            onChange={handleQuestionChange}
          />
          <button type="submit">Submit</button>
        </div>
      </form>
      <div>
        <h3>Tokens - {tokens}</h3>
        <h2>Answer:</h2>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
