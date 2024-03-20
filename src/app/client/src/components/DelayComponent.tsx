import React from 'react'

const DelayComponent = () => {
  const [show, setShow] = React.useState(false)

  React.useEffect(() => {
    const timeout = setTimeout(() => {
      setShow(true)
    }, 5000)

    return () => clearTimeout(timeout)

  }, [show])

  if (!show) return null

  return <>OK, Render</>
}

export default DelayComponent