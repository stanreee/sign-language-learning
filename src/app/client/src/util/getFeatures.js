const getFeatures = (landmarks) => {
    const features = landmarks.map((point) => [point.x, point.y, point.z]);
    return features;
}

export default getFeatures;