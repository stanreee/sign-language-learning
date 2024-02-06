// from https://github.com/tjallingt/react-youtube/tree/canary?tab=readme-ov-file
import YouTube, { YouTubeProps } from 'react-youtube';

type VideoClipProps = {
    letterVideoId: string;
};

const VideoClip = ({
    letterVideoId
  }: VideoClipProps) => {

  const onPlayerReady: YouTubeProps['onReady'] = (event) => {
    // access to player in all event handlers via event.target
    event.target.pauseVideo();
  }

  const opts: YouTubeProps['opts'] = {
    height: '234',
    width: '384',
    playerVars: {
      // https://developers.google.com/youtube/player_parameters
      autoplay: 0,
      cc_load_policy: 1,
      controls: 1,
    },
  };

  return (  
    <YouTube 
        videoId={letterVideoId}
        opts={opts} 
        onReady={onPlayerReady} />
  );
}

export default VideoClip