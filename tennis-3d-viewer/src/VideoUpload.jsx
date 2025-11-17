import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import './VideoUpload.css'

const API_URL = 'http://localhost:6000'

export function VideoUpload({ onVideoProcessed }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [socket, setSocket] = useState(null)
  const [outputFile, setOutputFile] = useState(null)
  const [framePreview, setFramePreview] = useState(null)
  const [frameInfo, setFrameInfo] = useState({ current: 0, total: 0 })

  useEffect(() => {
    // Connect to WebSocket
    const newSocket = io(API_URL)
    
    newSocket.on('connect', () => {
      console.log('Connected to server')
    })
    
    newSocket.on('processing_update', (data) => {
      console.log('Update:', data)
      setMessage(data.message)
      setProgress(data.progress)
      
      // Update frame preview if available
      if (data.frame_preview) {
        setFramePreview(data.frame_preview)
      }
      
      // Update frame info
      if (data.frame_num && data.total_frames) {
        setFrameInfo({ current: data.frame_num, total: data.total_frames })
      }
      
      if (data.status === 'complete') {
        setProcessing(false)
        setOutputFile(data.output_file)
        if (onVideoProcessed) {
          onVideoProcessed(data.output_file)
        }
      } else if (data.status === 'error') {
        setProcessing(false)
        setUploading(false)
      }
    })
    
    setSocket(newSocket)
    
    return () => newSocket.close()
  }, [onVideoProcessed])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setMessage(`Selected: ${selectedFile.name}`)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a video file')
      return
    }

    setUploading(true)
    setProcessing(true)
    setProgress(0)
    setMessage('Uploading video...')

    const formData = new FormData()
    formData.append('video', file)

    try {
      // Use XMLHttpRequest for upload progress tracking
      const xhr = new XMLHttpRequest()
      
      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percentComplete = Math.round((e.loaded / e.total) * 100)
          setProgress(percentComplete)
          setMessage(`Uploading: ${percentComplete}%`)
          console.log(`Upload progress: ${percentComplete}%`)
        }
      })

      // Handle completion
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText)
          if (data.success) {
            setSessionId(data.session_id)
            setMessage('Upload complete! Processing started...')
            setUploading(false)
            setProgress(0)
          } else {
            setMessage(`Error: ${data.error || 'Upload failed'}`)
            setUploading(false)
            setProcessing(false)
          }
        } else {
          setMessage(`Upload failed: Server returned ${xhr.status}`)
          setUploading(false)
          setProcessing(false)
        }
      })

      // Handle errors
      xhr.addEventListener('error', () => {
        setMessage('Upload failed: Network error')
        setUploading(false)
        setProcessing(false)
      })

      xhr.addEventListener('abort', () => {
        setMessage('Upload cancelled')
        setUploading(false)
        setProcessing(false)
      })

      // Send request
      xhr.open('POST', `${API_URL}/api/upload`)
      xhr.send(formData)

    } catch (error) {
      setMessage(`Upload failed: ${error.message}`)
      setUploading(false)
      setProcessing(false)
    }
  }

  const handleDownload = () => {
    if (outputFile) {
      window.open(`${API_URL}/api/download/${outputFile}`, '_blank')
    }
  }

  const handlePlayVideo = () => {
    if (outputFile) {
      const videoUrl = `${API_URL}/api/download/${outputFile}`
      // Create video element and play
      const videoPlayer = document.getElementById('result-video-player')
      if (videoPlayer) {
        videoPlayer.src = videoUrl
        videoPlayer.load()
        videoPlayer.play().catch(err => {
          console.error('Error playing video:', err)
          // Fallback to download
          handleDownload()
        })
      }
    }
  }

  const handleQuickProcess = async () => {
    // Process the local file directly (no upload needed)
    const localFilePath = 'copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov'
    
    setProcessing(true)
    setProgress(0)
    setMessage('Starting quick process...')

    try {
      const response = await fetch(`${API_URL}/api/process-local`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_path: localFilePath }),
      })

      const data = await response.json()

      if (data.success) {
        setSessionId(data.session_id)
        setMessage('Processing started! Watch the live preview below...')
      } else {
        setMessage(`Error: ${data.error}`)
        setProcessing(false)
      }
    } catch (error) {
      setMessage(`Failed: ${error.message}`)
      setProcessing(false)
    }
  }

  return (
    <div className="video-upload-container">
      <div className="upload-card">
        <h2>🎾 Upload Tennis Video</h2>
        
        <div className="file-input-wrapper">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            disabled={uploading || processing}
            id="video-input"
          />
          <label htmlFor="video-input" className="file-input-label">
            {file ? file.name : 'Choose Video File'}
          </label>
        </div>

        <div style={{ marginTop: '15px', color: 'rgba(255,255,255,0.7)', fontSize: '14px', textAlign: 'center' }}>
          💡 Tip: For faster processing, use smaller video files or shorter clips
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || uploading || processing}
          className="upload-button"
        >
          {uploading ? '⏫ Uploading...' : processing ? '⚙️ Processing...' : '🚀 Analyze Video'}
        </button>

        {/* Quick process button for local files */}
        <button
          onClick={handleQuickProcess}
          disabled={processing}
          className="upload-button"
          style={{ marginTop: '10px', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
        >
          ⚡ Quick Process (Local File)
        </button>

        {(uploading || processing) && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="progress-text">{progress}%</p>
          </div>
        )}

        {/* Live Frame Preview - LARGE AND PROMINENT */}
        {processing && (
          <div className="frame-preview" style={{
            marginTop: '30px',
            padding: '20px',
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '15px',
            border: '2px solid rgba(102, 126, 234, 0.5)'
          }}>
            <h3 style={{ color: 'white', marginBottom: '15px', textAlign: 'center' }}>
              🎬 Live Analysis Preview
            </h3>
            {framePreview ? (
              <>
                <img 
                  src={framePreview} 
                  alt="Processing..." 
                  style={{ 
                    width: '100%', 
                    borderRadius: '10px',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
                  }} 
                />
                <p className="frame-info" style={{
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  marginTop: '15px',
                  textAlign: 'center'
                }}>
                  Frame {frameInfo.current} / {frameInfo.total}
                </p>
              </>
            ) : (
              <div style={{
                padding: '60px',
                textAlign: 'center',
                color: 'rgba(255,255,255,0.6)',
                fontSize: '16px'
              }}>
                ⏳ Initializing analysis...
              </div>
            )}
          </div>
        )}

        {message && (
          <div className={`message ${processing ? 'processing' : ''}`}>
            {message}
          </div>
        )}

        {outputFile && (
          <div className="result-actions">
            <div className="video-player-container">
              <video 
                id="result-video-player"
                controls 
                autoPlay
                style={{ 
                  width: '100%', 
                  maxWidth: '800px', 
                  borderRadius: '10px',
                  marginBottom: '20px',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
                }}
              >
                <source src={`${API_URL}/api/download/${outputFile}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <button onClick={handlePlayVideo} className="play-button" style={{ marginRight: '10px' }}>
              ▶️ Play Video
            </button>
            <button onClick={handleDownload} className="download-button">
              ⬇️ Download Analyzed Video
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
