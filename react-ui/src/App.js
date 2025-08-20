import React, { useState } from 'react';
import axios from 'axios';
import jsPDF from 'jspdf';
import './App.css';
import { TextField, Button, Container, Typography, Paper, Grid } from '@mui/material';  // Material-UI components

function App() {
  const [prompt, setPrompt] = useState("");
  const [report, setReport] = useState("");
  const [similarReports, setSimilarReports] = useState([]);

  // Patient info
  const [patientName, setPatientName] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [dossierNumber, setDossierNumber] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [biopsyDate, setBiopsyDate] = useState("");

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:8000/generate", {
        doctor_prompt: prompt,
        patient_name: patientName,
        birth_date: birthDate,
        dossier_number: dossierNumber,
        doctor_name: doctorName,
        biopsy_date: biopsyDate,
      });

      setReport(res.data.report);
      setSimilarReports(res.data.similar_reports);
    } catch (error) {
      console.error("Error during request:", error);
      alert("An error occurred while generating the report.");
    }
  };

  const handleDownloadPDF = () => {
    const pdf = new jsPDF();
    const lines = pdf.splitTextToSize(report, 180); // Split text to fit page width

    let currentHeight = 10;  // Initial position for the first line on the page

    // Loop through the lines and add them to the PDF
    lines.forEach((line, index) => {
      // Check if the content exceeds the page height and add a new page if needed
      if (currentHeight + 10 > pdf.internal.pageSize.height) {
        pdf.addPage();
        currentHeight = 10; // Reset height for the new page
      }

      // Add the current line to the PDF at the specified position
      pdf.text(line, 10, currentHeight);
      currentHeight += 10; // Move down for the next line
    });

    // Save the generated PDF
    pdf.save("biopsy_report.pdf");
  };

  return (
    <Container maxWidth="lg" sx={{ padding: '2rem', fontFamily: 'Arial', backgroundColor: '#f4f7f8' }}>
      <Paper sx={{ padding: '2rem', backgroundColor: '#ffffff' }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#2d3e50' }}>Générateur de Rapport d'Anatomie Pathologique</Typography>
        
        <Grid container spacing={2} sx={{ marginTop: '1.5rem' }}>
          <Grid item xs={12} sm={6}>
            <TextField 
              label="Nom complet du patient" 
              variant="outlined" 
              fullWidth 
              value={patientName} 
              onChange={e => setPatientName(e.target.value)} 
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField 
              label="Date de naissance" 
              variant="outlined" 
              fullWidth 
              value={birthDate} 
              onChange={e => setBirthDate(e.target.value)} 
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField 
              label="Numéro de dossier" 
              variant="outlined" 
              fullWidth 
              value={dossierNumber} 
              onChange={e => setDossierNumber(e.target.value)} 
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField 
              label="Nom du médecin" 
              variant="outlined" 
              fullWidth 
              value={doctorName} 
              onChange={e => setDoctorName(e.target.value)} 
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField 
              label="Date de la biopsie" 
              variant="outlined" 
              fullWidth 
              value={biopsyDate} 
              onChange={e => setBiopsyDate(e.target.value)} 
            />
          </Grid>
        </Grid>

        <Typography variant="h6" sx={{ marginTop: '1.5rem', color: '#2d3e50' }}>Prompt du médecin</Typography>
        <TextField
          label="Décrivez les besoins de génération du rapport"
          multiline
          rows={6}
          variant="outlined"
          fullWidth
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
        />

        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleSubmit} 
          sx={{ marginTop: '1rem' }}
        >
          Générer le rapport
        </Button>

        <Typography variant="h6" sx={{ marginTop: '1.5rem', color: '#2d3e50' }}>Rapport généré (modifiable)</Typography>
        <TextField
          multiline
          rows={12}
          variant="outlined"
          fullWidth
          value={report}
          onChange={e => setReport(e.target.value)}
        />

        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleDownloadPDF} 
          sx={{ marginTop: '1rem' }}
        >
          Télécharger en PDF
        </Button>

        <Typography variant="h6" sx={{ marginTop: '1.5rem', color: '#2d3e50' }}>Rapports similaires</Typography>
        <ul>
          {similarReports.map(r => (
            <li key={r[0]}>{r[1]}</li>
          ))}
        </ul>
      </Paper>
    </Container>
  );
}

export default App;
