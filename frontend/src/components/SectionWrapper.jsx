import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

const variants = {
  hidden: { opacity: 0, y: 40 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] },
  },
};

export default function SectionWrapper({ children, id, className = '' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { threshold: 0.12 });

  return (
    <motion.section
      ref={ref}
      id={id}
      initial="hidden"
      animate={inView ? 'visible' : 'hidden'}
      variants={variants}
      className={`scroll-mt-20 ${className}`}
    >
      {children}
    </motion.section>
  );
}
