export default function SectionDivider({ label }) {
  return (
    <div className="flex items-center gap-5 my-16">
      <span className="flex-1 h-px bg-gradient-to-r from-transparent via-silver/30 to-transparent" />
      <span className="text-lg">{label || '🌿'}</span>
      <span className="flex-1 h-px bg-gradient-to-r from-transparent via-silver/30 to-transparent" />
    </div>
  );
}
