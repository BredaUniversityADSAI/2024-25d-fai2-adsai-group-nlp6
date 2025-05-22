# Emotion Classification Frontend

This React application provides a user interface for analyzing emotions in YouTube videos. Users can paste a YouTube URL, and the system will process the video, generate a transcript with emotion analysis, and visualize the emotional content through multiple interactive displays.

## Core Functionality

The application's primary purpose is to:
1. Accept a YouTube video URL from the user
2. Process the video using our backend emotion classification pipeline
3. Display the video alongside its transcript with emotion-based highlighting
4. Visualize emotional patterns and intensity throughout the video

## Data Structure

The backend provides emotion analysis in this format:

| Sentence | Start Time | End Time | Emotion | Sub Emotion | Intensity |
|----------|------------|----------|---------|-------------|-----------|
| "Hang on to your seats because Asia's next Top Model is back with a vengeance." | 00:00:01 | 00:00:05 | happiness | excitement | mild |
| "Do you want to be on top?" | 00:00:05 | 00:00:06 | happiness | curiosity | intense |
| "I am Filipino espresso." | 00:00:06 | 00:00:07 | neutral | neutral | intense |

This timestamped emotion data drives all visualizations and transcript highlighting in the UI.

## UI Layout: The Emotion Journey Experience

Our UI design takes inspiration from the concept of an "emotional journey," transforming technical analysis into an engaging visual experience:

### 1. **Horizon View** (Top)
- A minimalist, horizontally scrolling timeline representing the entire video
- Colored segments form an "emotional landscape" where mountains and valleys represent emotional intensity
- Current position indicated by a subtle pulse effect that ripples across the timeline

### 2. **Video & Transcript Fusion** (Center)
- The video player seamlessly integrates with an ambient background that subtly shifts color based on dominant emotions
- Beneath the video, the transcript flows like a conversation stream:
  - Each sentence appears in bubbles with gradient backgrounds matching their emotional tone
  - Size variations reflect intensity levels (larger = more intense)
  - A gentle animation brings the current sentence into focus as the video plays
  - Clicking any bubble instantly jumps to that moment in the video

### 3. **Emotion Pulse** (Right)
- A dynamic, real-time emotion display that breathes with the video's emotional rhythm
- Current emotions appear as glowing orbs in a constellation-like arrangement:
  - Primary emotions (happiness, sadness, anger) occupy central positions
  - Sub-emotions orbit around their primary emotions
  - Brightness and size reflect intensity and frequency
  - Transitions between emotions are smooth and fluid
- A minimal text overlay shows the exact emotion classification

### 4. **Video Memory** (Left)
- Previously analyzed videos appear as "memory cards" with emotional thumbnails
- Each card displays a tiny emotional timeline representing that video's overall emotional pattern
- A search bar with intelligent suggestions helps discover both new and past videos

### Interaction Philosophy
- **Fluid Navigation**: Every interaction feels smooth and intuitive, with gentle animations guiding the user experience
- **Immersive but Unobtrusive**: UI elements respond to the content but never distract from it
- **Emotional Context**: The interface itself subtly embodies the emotions it's displaying
- **Spatial Understanding**: Emotion patterns are presented spatially, making emotional arcs instantly understandable at a glance

All this complexity is presented through a clean, minimalist design language that feels familiar and easy to use. The visual complexity emerges organically from the data itself, not from complicated UI controls.

## User Flow

1. User enters or pastes a YouTube URL into the search bar
2. Application sends the URL to the backend for processing (showing a loading indicator)
3. Once processing completes (~30-60 seconds depending on video length):
   - Video loads in the player
   - Transcript appears below with emotion highlighting
   - Emotion visualizations populate in the right panel
4. User can:
   - Play/pause/seek in the video (transcript and emotion displays sync automatically)
   - Search within the transcript for specific content
   - Toggle between different emotion visualization tabs
   - Click on parts of the transcript to jump to that timestamp in the video

## Implementation Plan

### 1. Project Setup

```bash
# Create React application
npx create-react-app frontend
cd frontend

# Install dependencies
npm install axios chart.js react-chartjs-2 react-player @mui/material @emotion/react @emotion/styled framer-motion
```

### 2. Streamlined Directory Structure

```
frontend/
├── public/
├── src/
│   ├── components/        # All UI components
│   │   ├── UrlInput.js    # YouTube URL input/validation
│   │   ├── SearchBar.js   # Search functionality
│   │   ├── VideoHistory.js # Previous videos list
│   │   ├── VideoPlayer.js # YouTube player integration
│   │   ├── Transcript.js  # Emotion-highlighted transcript
│   │   ├── EmotionCurrent.js # Current emotion display
│   │   ├── EmotionBarChart.js # Emotion distribution chart
│   │   └── EmotionTimeline.js # Emotion intensity timeline
│   ├── api.js             # Backend API integration
│   ├── App.js             # Main application layout
│   ├── VideoContext.js    # Global state management
│   ├── utils.js           # Helper functions for emotion processing
│   └── index.js
└── package.json
```

### 3. Development Phases

#### Phase 1: YouTube Integration and Basic UI
- Create layout with three panels
- Implement YouTube URL input with validation
- Set up YouTube player component with basic controls
- Build history panel for previously analyzed videos

#### Phase 2: Transcript and Timeline
- Create transcript component with search functionality
- Implement timestamp synchronization between video and transcript
- Add emotion-based highlighting to transcript sentences

#### Phase 3: Emotion Visualizations
- Develop current emotion/sub-emotion display component
- Create emotion distribution bar chart
- Build emotion intensity timeline graph
- Connect all visualizations to video playback position

#### Phase 4: Backend Integration
- Implement API calls to analyze YouTube videos
- Create data parsers for the emotion analysis response
- Build loading and error states
- Connect UI components to the processed data

#### Phase 5: Refinements and Optimization
- Add transitions between emotion states
- Optimize performance for longer videos
- Implement caching for previously analyzed videos
- Add responsive design for different screen sizes

## API Integration

The frontend will connect to the backend through these key endpoints:

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Submit a YouTube video URL for analysis
export const analyzeVideo = async (youtubeUrl) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      url: youtubeUrl
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};

// Get analysis results for a specific video
export const getVideoAnalysis = async (videoId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/${videoId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching analysis:', error);
    throw error;
  }
};
```

## Data Processing

The frontend processes the backend emotion data to create visualizations:

```javascript
// Example of processing emotion data for visualization
const processEmotionData = (analysisData) => {
  // Group by emotion categories for the bar chart
  const emotionDistribution = analysisData.reduce((acc, item) => {
    acc[item.emotion] = (acc[item.emotion] || 0) + 1;
    return acc;
  }, {});

  // Process timeline data for the intensity graph
  const intensityTimeline = analysisData.map(item => ({
    time: item.start_time,
    emotion: item.emotion,
    intensity: item.intensity === 'mild' ? 0.3 :
              item.intensity === 'moderate' ? 0.6 : 0.9
  }));

  return { emotionDistribution, intensityTimeline };
};
```

## Running the Application

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

The application will be available at http://localhost:3000.

## Docker Integration

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Next Steps

1. Create initial proof-of-concept with YouTube URL input and basic player
2. Implement emotion-highlighted transcript component
3. Develop emotion visualization components
4. Connect to backend API and handle data flow
5. Add user experience enhancements and optimizations
