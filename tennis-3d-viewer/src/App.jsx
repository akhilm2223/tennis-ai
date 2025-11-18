import { useState, useRef, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Center } from '@react-three/drei'
import { CoachQ } from './CoachQ'
import { ModelInspector } from './ModelInspector'
import { VideoUpload } from './VideoUpload'
import { MentalCoach } from './MentalCoach'
import './App.css'

function App() {
  const [isTalking, setIsTalking] = useState(false)
  const [mouthOpenness, setMouthOpenness] = useState(0)
  const [currentMessage, setCurrentMessage] = useState('')
  const [showUpload, setShowUpload] = useState(false)
  const [showTraining, setShowTraining] = useState(false)
  const [showMentalCoach, setShowMentalCoach] = useState(false)
  const coachQRef = useRef()
  const animationFrameRef = useRef(null)

  // Backend API URL for mental coaching
  const COACH_API_URL = 'http://localhost:5002'

  const uploadTriggerPhrases = [
    'upload video',
    'upload a video',
    'analyze video',
    'analyze a video',
    'i want to upload',
    'want to upload'
  ]

  // Voice recognition setup
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.log('Speech recognition not supported')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    
    recognition.continuous = true
    recognition.interimResults = false
    recognition.lang = 'en-US'

    recognition.onresult = (event) => {
      // Ignore if Coach Q is talking (prevent hearing own speech)
      if (isTalking) {
        console.log('🔇 Ignoring - Coach Q is talking')
        return
      }
      
      const last = event.results.length - 1
      const text = event.results[last][0].transcript.toLowerCase().trim()
      
      console.log('Heard:', text)
      
      // Check for video-related keywords
      const videoKeywords = ['video', 'upload', 'analyze', 'open', 'show']
      const hasVideoKeyword = videoKeywords.some(keyword => text.includes(keyword))
      
      if (hasVideoKeyword) {
        console.log('🎬 Video interface requested!')
        handleUploadRequest()
        return // Exit early
      }
      
      // Check if user wants to upload video
      const wantsUpload = uploadTriggerPhrases.some(phrase => text.includes(phrase))
      
      if (wantsUpload) {
        console.log('🎬 Upload video requested!')
        handleUploadRequest()
        return // Exit early
      }
      
      // Default: activate coach for any speech with the actual question
      if (!isTalking) {
        handleAskCoach(text)
      }
    }

    recognition.onerror = (event) => {
      console.log('Speech recognition error:', event.error)
    }

    recognition.start()
    console.log('🎤 Voice recognition started - listening for your questions')

    return () => {
      recognition.stop()
    }
  }, [isTalking])

  const handleAskCoach = async (userQuestion = null) => {
    if (isTalking) return
    
    setIsTalking(true)
    
    try {
      // Use the user's actual question or a default greeting
      const query = userQuestion || 'Hello Coach, I need mental coaching advice'
      
      console.log('📝 Asking coach:', query)
      
      // Call backend Mental Coach API with RAG, Claude, and ElevenLabs TTS
      const response = await fetch(`${COACH_API_URL}/api/coach/chat-with-audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          session_id: 'voice-session-' + Date.now()
        })
      })
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Backend error:', errorText)
        throw new Error(`Failed to get coaching response: ${response.status}`)
      }
      
      // Check if we got audio or JSON text response
      const contentType = response.headers.get('content-type')
      console.log('Response content-type:', contentType)
      let message = ''
      let audioBlob = null
      
      if (contentType && contentType.includes('audio')) {
        // Got ElevenLabs audio!
        console.log('🎵 Received ElevenLabs audio')
        message = response.headers.get('X-Response-Text') || 'Audio response received'
        audioBlob = await response.blob()
        
        // Play ElevenLabs audio
        const audioUrl = URL.createObjectURL(audioBlob)
        const audio = new Audio(audioUrl)
        
        // Setup lip sync animation for audio playback
        const audioContext = new (window.AudioContext || window.webkitAudioContext)()
        const source = audioContext.createMediaElementSource(audio)
        const analyser = audioContext.createAnalyser()
        source.connect(analyser)
        analyser.connect(audioContext.destination)
        analyser.fftSize = 256
        const dataArray = new Uint8Array(analyser.frequencyBinCount)
        
        let isAnimating = true
        function animateFromAudio() {
          if (!isAnimating) return
          analyser.getByteFrequencyData(dataArray)
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length
          const openness = Math.min(average / 128, 1)
          setMouthOpenness(openness)
          requestAnimationFrame(animateFromAudio)
        }
        
        audio.onplay = () => {
          console.log("🎤 Coach Q started talking (ElevenLabs)")
          setCurrentMessage('')  // Don't show text, just audio
          isAnimating = true
          animateFromAudio()
        }
        
        audio.onended = () => {
          console.log("🤫 Coach Q finished talking")
          isAnimating = false
          setMouthOpenness(0)
          setIsTalking(false)
          setTimeout(() => setCurrentMessage(''), 1000)
          URL.revokeObjectURL(audioUrl)
        }
        
        audio.play()
        return // Exit early, audio is playing
      } else {
        // No ElevenLabs audio available
        console.error('ElevenLabs audio not available')
        setIsTalking(false)
        throw new Error('ElevenLabs audio not available. Please check server configuration.')
      }
      
    } catch (error) {
      console.error('Error calling coach API:', error)
      setIsTalking(false)
      setMouthOpenness(0)
      
      // Fallback message
      const fallbackMsg = "I'm having trouble connecting to the coaching service. Please make sure the backend is running."
      const fallbackUtterance = new SpeechSynthesisUtterance(fallbackMsg)
      fallbackUtterance.rate = 0.9
      fallbackUtterance.pitch = 1.1
      fallbackUtterance.onend = () => {
        setIsTalking(false)
        setCurrentMessage('')
        setMouthOpenness(0)
      }
      window.speechSynthesis.speak(fallbackUtterance)
    }
  }

  const handleUploadRequest = () => {
    // Open video upload interface directly
    setShowUpload(true)
  }

  return (
    <div className="canvas-container">
      {/* 3D Scene - fades out when upload, training, or mental coach is shown */}
      <div style={{
        opacity: (showUpload || showTraining || showMentalCoach) ? 0 : 1,
        transform: (showUpload || showTraining || showMentalCoach) ? 'scale(2)' : 'scale(1)',
        transition: 'all 1s ease-in-out',
        width: '100%',
        height: '100%',
        pointerEvents: (showUpload || showTraining || showMentalCoach) ? 'none' : 'auto'
      }}>
        <Canvas 
          camera={{ position: [0, 0, 3], fov: 75 }}
          gl={{ alpha: false }}
          style={{ background: '#000000' }}
        >
          <color attach="background" args={['#000000']} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          <directionalLight position={[-5, 5, -5]} intensity={0.4} />
          <pointLight position={[0, 2, 2]} intensity={0.5} />
          <Center>
            <CoachQ ref={coachQRef} isTalking={isTalking} mouthOpenness={mouthOpenness} />
            <ModelInspector />
          </Center>
          <OrbitControls 
            enableZoom={true} 
            minDistance={1.5}
            maxDistance={10}
            target={[0, 0, 0]}
          />
        </Canvas>
      </div>
      
      {/* MindServe Logo - Clean & Minimal */}
      {!showUpload && !showTraining && !showMentalCoach && (
        <div style={{
          position: 'fixed',
          top: '40px',
          left: '50%',
          transform: 'translateX(-50%)',
          textAlign: 'center',
          zIndex: 1000
        }}>
          <h1 style={{
            fontSize: 'clamp(32px, 5vw, 56px)',
            fontWeight: '700',
            color: '#ffffff',
            margin: 0,
            padding: '10px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            letterSpacing: '-1px'
          }}>
            MindServe
          </h1>
          <p style={{
            color: 'rgba(255, 255, 255, 0.6)',
            fontSize: 'clamp(13px, 1.8vw, 16px)',
            margin: '8px 0 0 0',
            fontWeight: '400',
            letterSpacing: '0.5px',
            textTransform: 'uppercase'
          }}>
            Your Tennis Coach
          </p>
        </div>
      )}

      {/* Speech text - bottom of screen */}
      {currentMessage && (
        <div style={{
          position: 'fixed',
          bottom: '120px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,240,255,0.95) 100%)',
          color: '#333',
          padding: '20px 40px',
          borderRadius: '20px',
          fontSize: 'clamp(16px, 2.5vw, 24px)',
          fontWeight: '500',
          width: '90%',
          maxWidth: '800px',
          textAlign: 'center',
          boxShadow: '0 10px 40px rgba(0,0,0,0.3), 0 0 0 3px rgba(102, 126, 234, 0.3)',
          zIndex: 9999,
          animation: 'fadeIn 0.3s ease-in',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          lineHeight: '1.5'
        }}>
          💬 {currentMessage}
        </div>
      )}

      {/* Navigation Buttons - Clean UI */}
      {!showUpload && !showTraining && !showMentalCoach && (
        <div style={{
          position: 'fixed',
          bottom: '40px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: '15px',
          zIndex: 1000
        }}>
          <button
            onClick={() => setShowUpload(true)}
            style={{
              padding: '16px 32px',
              fontSize: '16px',
              fontWeight: '600',
              background: 'rgba(255, 255, 255, 0.95)',
              color: '#1a1a1a',
              border: '1px solid rgba(0, 0, 0, 0.1)',
              borderRadius: '12px',
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              transition: 'all 0.2s ease',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 1)'
              e.target.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)'
              e.target.style.transform = 'translateY(-2px)'
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 0.95)'
              e.target.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)'
              e.target.style.transform = 'translateY(0)'
            }}
          >
            Video Analysis
          </button>
          
          <button
            onClick={() => setShowMentalCoach(true)}
            style={{
              padding: '16px 32px',
              fontSize: '16px',
              fontWeight: '600',
              background: 'rgba(255, 255, 255, 0.95)',
              color: '#1a1a1a',
              border: '1px solid rgba(0, 0, 0, 0.1)',
              borderRadius: '12px',
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              transition: 'all 0.2s ease',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 1)'
              e.target.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)'
              e.target.style.transform = 'translateY(-2px)'
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 0.95)'
              e.target.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)'
              e.target.style.transform = 'translateY(0)'
            }}
          >
            Mental Coaching
          </button>
        </div>
      )}

      {/* Video Upload Modal */}
      {showUpload && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          animation: 'zoomIn 1s ease-out',
          width: '100%',
          height: '100%',
          zIndex: 2000
        }}>
          <VideoUpload onVideoProcessed={(file) => console.log('Processed:', file)} />
          
          <button 
            onClick={() => setShowUpload(false)}
            style={{
              position: 'fixed',
              top: '20px',
              right: '20px',
              padding: '10px 20px',
              fontSize: '16px',
              fontWeight: 'bold',
              background: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              border: '2px solid white',
              borderRadius: '50px',
              cursor: 'pointer',
              zIndex: 3000,
              backdropFilter: 'blur(10px)'
            }}
          >
            ✕ Close
          </button>
        </div>
      )}

      {/* Mental Coaching Modal */}
      {showMentalCoach && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          animation: 'zoomIn 1s ease-out',
          width: '100%',
          height: '100%',
          zIndex: 2000
        }}>
          <MentalCoach />
          
          <button 
            onClick={() => setShowMentalCoach(false)}
            style={{
              position: 'fixed',
              top: '20px',
              right: '20px',
              padding: '10px 20px',
              fontSize: '16px',
              fontWeight: 'bold',
              background: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              border: '2px solid white',
              borderRadius: '50px',
              cursor: 'pointer',
              zIndex: 3000,
              backdropFilter: 'blur(10px)'
            }}
          >
            ✕ Close
          </button>
        </div>
      )}

    </div>
  )
}

export default App
