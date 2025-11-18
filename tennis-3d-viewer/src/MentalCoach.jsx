import { useState, useRef, useEffect, useCallback } from 'react'
import './MentalCoach.css'

const COACH_API_URL = 'http://localhost:5002'

export function MentalCoach() {
  const [status, setStatus] = useState('ready') // ready, recording, processing
  const [transcript, setTranscript] = useState('')
  const [response, setResponse] = useState('')
  const [error, setError] = useState('')
  const [timer, setTimer] = useState('')
  const [audioUrl, setAudioUrl] = useState(null)
  const [showTranscript, setShowTranscript] = useState(false)
  const [showResponse, setShowResponse] = useState(false)
  const [showAudioPlayer, setShowAudioPlayer] = useState(false)
  const [isCoachSpeaking, setIsCoachSpeaking] = useState(false)
  const [audioVisualizerData, setAudioVisualizerData] = useState(Array(20).fill(0))

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const recordingStartTimeRef = useRef(null)
  const timerIntervalRef = useRef(null)
  const audioElementRef = useRef(null)
  const audioUrlRef = useRef(null)
  const analyzerRef = useRef(null)
  const animationFrameRef = useRef(null)

  const handleRecordingStop = useCallback(async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
    
    const formData = new FormData()
    formData.append('audio', audioBlob, 'recording.wav')

    try {
      const response = await fetch(`${COACH_API_URL}/api/coach/voice-chat`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        let errorMessage = 'Failed to get response from coach'
        try {
          const errorData = await response.json()
          errorMessage = errorData.error || errorMessage
        } catch (e) {
          // If response is not JSON, use status text
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      // Get transcribed text from headers
      const transcribedText = response.headers.get('X-Transcribed-Text')
      const responseTextHeader = response.headers.get('X-Response-Text')

      console.log('=== COACH RESPONSE DEBUG ===')
      console.log('Transcribed Text:', transcribedText)
      console.log('Response Text (full):', responseTextHeader)
      console.log('Response Text Length:', responseTextHeader?.length || 0)
      console.log('All Response Headers:')
      for (let [key, value] of response.headers.entries()) {
        console.log(`  ${key}: ${value}`)
      }
      console.log('===========================')

      if (transcribedText) {
        setTranscript(transcribedText)
        setShowTranscript(true)
      }

      if (responseTextHeader) {
        setResponse(responseTextHeader)
        setShowResponse(true)
      } else {
        console.warn('⚠️ No X-Response-Text header found in response')
      }

      // Get audio blob
      const audioBlob = await response.blob()
      const url = URL.createObjectURL(audioBlob)
      setAudioUrl(url)
      audioUrlRef.current = url
      setShowAudioPlayer(true)
      
      console.log('Audio response received, setting up playback...')
      
      setError('')

    } catch (error) {
      console.error('Error calling coach API:', error)
      let errorMessage = error.message
      
      // Provide more helpful error messages
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessage = 'Cannot connect to coach server. Make sure the voice coach server is running on port 5002.'
      } else if (error.message.includes('503')) {
        errorMessage = 'Coach service unavailable. Check if all required services (TTS, STT) are configured.'
      }
      
      setError(errorMessage)
      setStatus('ready')
    }
  }, [])

  // Request microphone access on mount
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream)
        mediaRecorder.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data)
        }
        mediaRecorder.onstop = handleRecordingStop
        mediaRecorderRef.current = mediaRecorder
      })
      .catch(err => {
        setError('Microphone access denied. Please allow microphone access to use this app.')
        setStatus('ready')
      })

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
      }
      if (audioElementRef.current) {
        audioElementRef.current.pause()
        audioElementRef.current = null
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
      }
    }
  }, [handleRecordingStop])

  // Handle audio playback when audioUrl is set
  useEffect(() => {
    if (!audioUrl) return
    
    const setupAudio = async () => {
      if (!audioElementRef.current) return
      
      try {
        audioElementRef.current.src = audioUrl
        await audioElementRef.current.load()
        
        // Create audio context for visualization
        const audioContext = new (window.AudioContext || window.webkitAudioContext)()
        const source = audioContext.createMediaElementAudioSource(audioElementRef.current)
        const analyzer = audioContext.createAnalyser()
        analyzer.fftSize = 256
        source.connect(analyzer)
        analyzer.connect(audioContext.destination)
        analyzerRef.current = analyzer
        
        // Play audio
        await audioElementRef.current.play()
        console.log('Audio playing automatically')
        setIsCoachSpeaking(true)
        setStatus('ready') // Set to ready when audio starts playing
        updateVisualizer()
        
        // Set up ended handler
        audioElementRef.current.onended = () => {
          setIsCoachSpeaking(false)
          if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current)
          }
        }
      } catch (err) {
        console.warn('Audio setup/playback error:', err)
        setStatus('ready') // Set to ready even if autoplay fails
      }
    }
    
    setupAudio()
  }, [audioUrl])

  // Update visualizer animation
  const updateVisualizer = () => {
    if (analyzerRef.current && audioElementRef.current && !audioElementRef.current.paused) {
      const dataArray = new Uint8Array(analyzerRef.current.frequencyBinCount)
      analyzerRef.current.getByteFrequencyData(dataArray)
      
      // Sample every nth value to get 20 bars
      const step = Math.floor(dataArray.length / 20)
      const sampledData = []
      for (let i = 0; i < 20; i++) {
        sampledData.push(dataArray[i * step] / 255)
      }
      setAudioVisualizerData(sampledData)
      animationFrameRef.current = requestAnimationFrame(updateVisualizer)
    }
  }

  const startRecording = () => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'recording') {
      return
    }

    audioChunksRef.current = []
    recordingStartTimeRef.current = Date.now()
    
    // Stop any currently playing audio
    if (audioElementRef.current && !audioElementRef.current.paused) {
      audioElementRef.current.pause()
      audioElementRef.current.currentTime = 0
    }
    
    mediaRecorderRef.current.start()
    
    setStatus('recording')
    setError('')
    setShowTranscript(false)
    setShowResponse(false)
    setShowAudioPlayer(false)
    setTranscript('')
    setResponse('')

    startTimer()
  }

  const stopRecording = () => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
      return
    }

    mediaRecorderRef.current.stop()
    setStatus('processing')
    clearTimer()
  }

  const startTimer = () => {
    setTimer('0s')
    timerIntervalRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartTimeRef.current) / 1000)
      setTimer(`${elapsed}s`)
    }, 100)
  }

  const clearTimer = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
    setTimer('')
  }


  const toggleRecording = () => {
    if (status === 'recording') {
      stopRecording()
    } else {
      startRecording()
    }
  }

  return (
    <div className="mental-coach-container">
      <div className="mental-coach-card">
        <h1>🎾 Tennis Mental Coach</h1>
        <p className="subtitle">Voice-powered mental performance coaching</p>
        
        <div className={`status ${status}`}>
          {status === 'ready' && 'Ready'}
          {status === 'recording' && 'Recording...'}
          {status === 'processing' && 'Processing...'}
        </div>

        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          marginTop: '20px',
          marginBottom: '20px'
        }}>
          <button 
            className={`mic-button ${status === 'recording' ? 'recording' : ''}`}
            onClick={toggleRecording}
            disabled={status === 'processing'}
            title={status === 'recording' ? 'Click to stop recording' : 'Click to start recording'}
            style={{
              width: '120px',
              height: '120px',
              borderRadius: '50%',
              border: 'none',
              background: status === 'recording' 
                ? 'linear-gradient(135deg, #f44336 0%, #e91e63 100%)'
                : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              fontSize: '50px',
              cursor: status === 'processing' ? 'not-allowed' : 'pointer',
              margin: '30px auto',
              transition: 'all 0.3s ease',
              boxShadow: status === 'recording'
                ? '0 10px 30px rgba(244, 67, 54, 0.4)'
                : '0 10px 30px rgba(102, 126, 234, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: status === 'processing' ? 0.6 : 1
            }}
          >
            {status === 'recording' ? '⏹️' : '🎤'}
          </button>
        </div>
        
        {status === 'ready' && (
          <div style={{
            marginTop: '10px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#667eea'
          }}>
            Click to Record
          </div>
        )}
        
        {status === 'recording' && (
          <div style={{
            marginTop: '10px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#f44336'
          }}>
            Recording... Click to Stop
          </div>
        )}

        {timer && <div className="timer">{timer}</div>}

        {showTranscript && (
          <div className="transcript show">
            <div className="transcript-label">Your Question:</div>
            <div>{transcript}</div>
          </div>
        )}

        {showResponse && (
          <div className={`response show ${isCoachSpeaking ? 'speaking' : ''}`}>
            <div className="response-label">
              {isCoachSpeaking ? '🗣️ Coach is speaking...' : "Coach's Response:"}
            </div>
            <div>{response}</div>
          </div>
        )}

        {showAudioPlayer && audioUrl && (
          <div className="audio-player show">
            {/* Audio Visualizer */}
            {isCoachSpeaking && (
              <div className="audio-visualizer">
                {audioVisualizerData.map((value, index) => (
                  <div
                    key={index}
                    className="visualizer-bar"
                    style={{
                      height: `${Math.max(5, value * 100)}%`,
                      animation: isCoachSpeaking ? 'none' : 'none'
                    }}
                  />
                ))}
              </div>
            )}
            
            <audio 
              ref={audioElementRef}
              controls
              autoPlay
              preload="auto"
            />
          </div>
        )}
        
        {/* Hidden audio element for autoplay when showAudioPlayer is false */}
        {!showAudioPlayer && (
          <audio 
            ref={audioElementRef}
            style={{ display: 'none' }}
            autoPlay
            preload="auto"
          />
        )}

        {error && (
          <div className="error show">
            {error}
          </div>
        )}

        {status === 'ready' && !showResponse && (
          <div className="info">
            💡 Click the microphone button above to start recording. Speak your question, then click again to stop and get your response.
          </div>
        )}
      </div>
    </div>
  )
}

